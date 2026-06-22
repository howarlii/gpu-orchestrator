"""Task queue + dispatcher for single-node multi-GPU.

Tasks are single shell commands. The dispatcher assigns queued tasks to GPUs
under two live-tunable constraints: max tasks per GPU and a minimum free-HBM
threshold. Supports priority preemption, forced GPU evacuation and requeue.

State is persisted in SQLite so the panel survives restarts. Running PIDs are
tracked in-memory; on restart, previously-running tasks are marked 'lost'.
"""
from __future__ import annotations

import json
import os
import re
import signal
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

import psutil

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "orchestrator.db"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

ACTIVE = ("queued", "running")


def _now() -> float:
    return time.time()


class Scheduler:
    def __init__(self, monitor) -> None:
        self.monitor = monitor
        self.lock = threading.RLock()
        self.procs: dict[int, subprocess.Popen] = {}  # task_id -> Popen
        # tasks re-adopted after a restart: task_id -> pid (not our children,
        # so they can only be polled by PID existence, not waitpid()'d)
        self.orphans: dict[int, int] = {}
        # revision counter: bumped on every task/config mutation so the server
        # can avoid re-broadcasting the (heavy) task list when nothing changed.
        self.rev = 0
        # timestamp of the most recent dispatch, for the optional cooldown gate
        self._last_launch_ts = 0.0
        self.db = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init_db()
        self._recover()

    def _bump(self) -> None:
        """Mark task/config state as changed (drives server-side diffing)."""
        self.rev += 1

    def get_rev(self) -> int:
        return self.rev

    # ---- schema -------------------------------------------------------------
    def _init_db(self) -> None:
        with self.lock:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    command TEXT NOT NULL,
                    params TEXT DEFAULT '',
                    priority INTEGER DEFAULT 0,
                    num_gpus INTEGER DEFAULT 1,
                    min_free_hbm_gb REAL,
                    status TEXT DEFAULT 'queued',
                    gpu_ids TEXT DEFAULT '',
                    pid INTEGER,
                    exit_code INTEGER,
                    created_at REAL,
                    started_at REAL,
                    ended_at REAL,
                    log_path TEXT
                );
                CREATE TABLE IF NOT EXISTS config (k TEXT PRIMARY KEY, v TEXT);
                """
            )
            self.db.commit()
            defaults = {
                "max_tasks_per_gpu": "1",
                "min_free_hbm_gb": "10",
                "min_free_ram_gb": "0",       # 0 = no RAM gate
                "max_concurrent_tasks": "0",  # 0 = stop dispatching (pause)
                "dispatch_cooldown_s": "0",   # 0 = no cooldown between launches
                "reserved_gpus": "[]",
                "paused": "0",
            }
            for k, v in defaults.items():
                self.db.execute(
                    "INSERT OR IGNORE INTO config(k, v) VALUES(?, ?)", (k, v))
            self.db.commit()

    def _recover(self) -> None:
        """On restart, re-adopt still-running detached processes by PID.

        Children were launched with setsid(), so they survive this process'
        death. We can't waitpid() a non-child, so we poll PID existence in
        reap(). Tasks whose process is gone are marked 'lost'.
        """
        with self.lock:
            rows = self.db.execute(
                "SELECT id, pid, started_at FROM tasks "
                "WHERE status='running'").fetchall()
            for r in rows:
                if r["pid"] and self._pid_alive(r["pid"], r["started_at"]):
                    self.orphans[r["id"]] = r["pid"]
                else:
                    self.db.execute(
                        "UPDATE tasks SET status='lost', ended_at=? WHERE id=?",
                        (_now(), r["id"]))
            self.db.commit()

    @staticmethod
    def _pid_alive(pid: int, started_at: Optional[float]) -> bool:
        """True if pid exists and (when known) started before the task did,
        guarding against PID reuse by an unrelated process."""
        try:
            p = psutil.Process(pid)
            if started_at and p.create_time() > started_at + 5:
                return False
            return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
        except Exception:
            return False

    # ---- config -------------------------------------------------------------
    def get_config(self) -> dict:
        with self.lock:
            rows = self.db.execute("SELECT k, v FROM config").fetchall()
        cfg = {r["k"]: r["v"] for r in rows}
        return {
            "max_tasks_per_gpu": int(cfg.get("max_tasks_per_gpu", 1)),
            "min_free_hbm_gb": float(cfg.get("min_free_hbm_gb", 10)),
            "min_free_ram_gb": float(cfg.get("min_free_ram_gb", 0)),
            "max_concurrent_tasks": int(cfg.get("max_concurrent_tasks", 0)),
            "dispatch_cooldown_s": float(cfg.get("dispatch_cooldown_s", 0)),
            "reserved_gpus": json.loads(cfg.get("reserved_gpus", "[]")),
            "paused": cfg.get("paused", "0") == "1",
        }

    def set_config(self, **kw) -> dict:
        with self.lock:
            for k, v in kw.items():
                if k == "reserved_gpus":
                    v = json.dumps(sorted(set(int(x) for x in v)))
                elif k == "paused":
                    v = "1" if v else "0"
                else:
                    v = str(v)
                self.db.execute(
                    "INSERT INTO config(k, v) VALUES(?, ?) "
                    "ON CONFLICT(k) DO UPDATE SET v=excluded.v", (k, v))
            self.db.commit()
            self._bump()
        return self.get_config()

    # ---- task CRUD ----------------------------------------------------------
    def add_task(self, command: str, name: str = "", priority: int = 0,
                 num_gpus: int = 1, min_free_hbm_gb: Optional[float] = None,
                 params: str = "") -> int:
        with self.lock:
            cur = self.db.execute(
                "INSERT INTO tasks(name, command, params, priority, num_gpus, "
                "min_free_hbm_gb, status, created_at) "
                "VALUES(?,?,?,?,?,?,'queued',?)",
                (name or command[:40], command, params, priority,
                 max(1, num_gpus), min_free_hbm_gb, _now()))
            self.db.commit()
            self._bump()
            return cur.lastrowid

    def list_tasks(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute(
                "SELECT * FROM tasks ORDER BY "
                "CASE status WHEN 'running' THEN 0 WHEN 'queued' THEN 1 "
                "ELSE 2 END, priority DESC, id ASC").fetchall()
        return [dict(r) for r in rows]

    def update_tasks(self, ids: list[int], **fields) -> None:
        if not ids or not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values())
        with self.lock:
            for tid in ids:
                self.db.execute(f"UPDATE tasks SET {cols} WHERE id=?",
                                vals + [tid])
            self.db.commit()
            self._bump()

    def delete_tasks(self, ids: list[int]) -> None:
        for tid in ids:
            self._kill(tid, status="killed")
        with self.lock:
            self.db.executemany("DELETE FROM tasks WHERE id=?",
                                [(t,) for t in ids])
            self.db.commit()
            self._bump()

    def set_status(self, ids: list[int], status: str) -> None:
        """Manually reclassify a terminal task (e.g. a 'lost' task -> done/
        failed). Never touches running/queued tasks."""
        if status not in ("done", "failed") or not ids:
            return
        with self.lock:
            for tid in ids:
                self.db.execute(
                    "UPDATE tasks SET status=? WHERE id=? AND status IN "
                    "('lost', 'done', 'failed', 'killed')", (status, tid))
            self.db.commit()
            self._bump()

    def requeue_tasks(self, ids: list[int]) -> None:
        for tid in ids:
            self._kill(tid, status="queued")
        with self.lock:
            for tid in ids:
                self.db.execute(
                    "UPDATE tasks SET status='queued', gpu_ids='', pid=NULL, "
                    "started_at=NULL, ended_at=NULL, exit_code=NULL "
                    "WHERE id=? AND status NOT IN ('running')", (tid,))
            self.db.commit()
            self._bump()

    def retry_failed(self) -> list[int]:
        """Requeue every failed/lost task (one-click retry)."""
        with self.lock:
            rows = self.db.execute(
                "SELECT id FROM tasks WHERE status IN ('failed', 'lost')"
            ).fetchall()
        ids = [r["id"] for r in rows]
        self.requeue_tasks(ids)
        return ids

    # ---- process control ----------------------------------------------------
    def _kill(self, tid: int, status: str) -> None:
        with self.lock:
            proc = self.procs.pop(tid, None)
            orphan_pid = self.orphans.pop(tid, None)
        # re-adopted task: kill its process group directly by PID
        if orphan_pid and self._pid_alive(orphan_pid, None):
            try:
                os.killpg(os.getpgid(orphan_pid), signal.SIGTERM)
            except Exception:
                pass
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                pass
            for _ in range(20):
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        with self.lock:
            row = self.db.execute(
                "SELECT status FROM tasks WHERE id=?", (tid,)).fetchone()
            if row and row["status"] == "running":
                self.db.execute(
                    "UPDATE tasks SET status=?, ended_at=? WHERE id=?",
                    (status, _now(), tid))
                self.db.commit()
                self._bump()

    def _launch(self, task: dict, gpu_ids: list[int]) -> None:
        # each run gets its own log file: task_<id>.<run>.log (run starts at 1)
        runs = self._run_logs(task["id"])
        run_no = (max(runs) + 1) if runs else 1
        log_path = str(LOG_DIR / f"task_{task['id']}.{run_no}.log")
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        logf = open(log_path, "wb", buffering=0)
        logf.write(f"# task {task['id']} on GPU {gpu_ids} @ "
                   f"{time.ctime()}\n# {task['command']}\n\n".encode())
        proc = subprocess.Popen(
            task["command"], shell=True, env=env,
            stdout=logf, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid, cwd=os.path.expanduser("~"))
        with self.lock:
            self.procs[task["id"]] = proc
            self.db.execute(
                "UPDATE tasks SET status='running', gpu_ids=?, pid=?, "
                "started_at=?, log_path=? WHERE id=?",
                (json.dumps(gpu_ids), proc.pid, _now(), log_path, task["id"]))
            self.db.commit()
            self._bump()
            self._last_launch_ts = _now()

    def evacuate_gpu(self, gpu: int, reserve: bool = True,
                     kill: bool = True) -> None:
        """Reserve a GPU so no new tasks land on it. When ``kill`` is true,
        also kill+requeue every task currently running on it; when false,
        running tasks are left alone (drain only — just stop new dispatch)."""
        if kill:
            with self.lock:
                rows = self.db.execute(
                    "SELECT id, gpu_ids FROM tasks WHERE status='running'"
                ).fetchall()
            victims = [r["id"] for r in rows
                       if gpu in json.loads(r["gpu_ids"] or "[]")]
            self.requeue_tasks(victims)
        if reserve:
            cfg = self.get_config()
            res = set(cfg["reserved_gpus"]) | {gpu}
            self.set_config(reserved_gpus=list(res))

    def run_now(self, tid: int, sample: dict) -> bool:
        """Force-launch a queued task immediately on the GPU(s) with the most
        free HBM, bypassing the per-GPU cap, HBM gate and dispatch cooldown.
        Reserved GPUs are still avoided. Returns True if it launched."""
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM tasks WHERE id=?", (tid,)).fetchone()
        if not row or row["status"] != "queued":
            return False
        task = dict(row)
        need = task["num_gpus"]
        cfg = self.get_config()
        reserved = set(cfg["reserved_gpus"])
        ngpu = self.monitor.count or 0
        cands = [g for g in range(ngpu) if g not in reserved]
        if len(cands) < need:
            return False
        cands.sort(key=lambda g: -self._free_hbm_gb(g, sample))
        self._launch(task, cands[:need])
        return True

    def task_usage(self, sample: dict) -> dict:
        """Per running task: live GPU-mem / SM (matched from the sample's GPU
        processes by walking each task's process tree) plus CPU / RAM. Includes
        running tasks that have not yet allocated any GPU memory."""
        with self.lock:
            rows = self.db.execute(
                "SELECT id, pid, gpu_ids FROM tasks WHERE status='running'"
            ).fetchall()
        # pid -> list of (gpu_index, proc) from the current sample
        proc_by_pid: dict[int, list] = {}
        for g in sample.get("gpus", []):
            for pr in g.get("procs", []):
                proc_by_pid.setdefault(pr["pid"], []).append((g["index"], pr))
        out: dict[int, dict] = {}
        for r in rows:
            pid = r["pid"]
            if not pid:
                continue
            try:
                p = psutil.Process(pid)
                tree = [pid] + [c.pid for c in p.children(recursive=True)]
            except Exception:
                tree = [pid]
            gpu_mem = sm = 0
            cpu = 0.0
            rss = 0
            gpus_used: set[int] = set()
            matched = False
            for x in tree:
                for gi, pr in proc_by_pid.get(x, []):
                    matched = True
                    gpu_mem += pr.get("gpu_mem") or 0
                    sm += pr.get("sm") or 0
                    cpu += pr.get("cpu") or 0.0
                    rss += pr.get("rss") or 0
                    gpus_used.add(gi)
            if not matched:
                # not on any GPU yet: sample the task's own process tree
                cpu, rss = self.monitor._cpu_ram(pid)
            out[r["id"]] = {
                "pid": pid,
                "gpu_mem": gpu_mem,
                "sm": sm,
                "cpu": round(cpu, 1),
                "rss": rss,
                "on_gpu": matched,
                "gpus": sorted(gpus_used) or json.loads(r["gpu_ids"] or "[]"),
            }
        return out

    # ---- dispatch loop ------------------------------------------------------
    def _free_hbm_gb(self, gpu: int, sample: dict) -> float:
        for g in sample.get("gpus", []):
            if g["index"] == gpu and g.get("mem_used") is not None:
                return (g["mem_total"] - g["mem_used"]) / 1024**3
        return 0.0

    def _running_count(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        with self.lock:
            rows = self.db.execute(
                "SELECT gpu_ids FROM tasks WHERE status='running'").fetchall()
        for r in rows:
            for g in json.loads(r["gpu_ids"] or "[]"):
                counts[g] = counts.get(g, 0) + 1
        return counts

    def _running_tasks(self) -> int:
        with self.lock:
            row = self.db.execute(
                "SELECT COUNT(*) AS n FROM tasks WHERE status='running'"
            ).fetchone()
        return row["n"] if row else 0

    def reap(self) -> bool:
        """Collect finished processes. Returns True if anything changed."""
        changed = False
        with self.lock:
            items = list(self.procs.items())
        for tid, proc in items:
            rc = proc.poll()
            if rc is None:
                continue
            with self.lock:
                self.procs.pop(tid, None)
                self.db.execute(
                    "UPDATE tasks SET status=?, exit_code=?, ended_at=? "
                    "WHERE id=?",
                    ("done" if rc == 0 else "failed", rc, _now(), tid))
                self.db.commit()
                self._bump()
            changed = True
        # re-adopted orphans: we can't get an exit code, only liveness
        with self.lock:
            orphans = list(self.orphans.items())
        for tid, pid in orphans:
            if self._pid_alive(pid, None):
                continue
            with self.lock:
                self.orphans.pop(tid, None)
                self.db.execute(
                    "UPDATE tasks SET status='done', ended_at=? "
                    "WHERE id=? AND status='running'", (_now(), tid))
                self.db.commit()
                self._bump()
            changed = True
        return changed

    def tick(self, sample: dict) -> bool:
        """One dispatch step. Returns True if state changed."""
        changed = self.reap()
        cfg = self.get_config()
        if cfg["paused"]:
            return changed
        # global RAM gate: only dispatch while free RAM stays above threshold
        min_free_ram = cfg["min_free_ram_gb"]
        if min_free_ram > 0:
            try:
                avail = psutil.virtual_memory().available / 1024**3
            except Exception:
                avail = float("inf")
            if avail < min_free_ram:
                return changed

        # optional dispatch cooldown: after a launch, hold off dispatching the
        # next task so a freshly-started task has time to actually claim HBM
        # (otherwise its GPU still looks free and we over-allocate it).
        cooldown = cfg["dispatch_cooldown_s"]
        if cooldown > 0 and (_now() - self._last_launch_ts) < cooldown:
            return changed

        # max_concurrent_tasks == 0 means "stop dispatching" (this is the pause
        # control now that the explicit pause checkbox is gone); a positive
        # value is a hard cap on total running tasks.
        if cfg["max_concurrent_tasks"] == 0:
            return changed

        reserved = set(cfg["reserved_gpus"])
        max_per = cfg["max_tasks_per_gpu"]
        max_concurrent = cfg["max_concurrent_tasks"]
        counts = self._running_count()
        running_total = self._running_tasks()
        ngpu = self.monitor.count or 0

        with self.lock:
            queued = self.db.execute(
                "SELECT * FROM tasks WHERE status='queued' "
                "ORDER BY priority DESC, id ASC").fetchall()
        queued = [dict(r) for r in queued]

        for task in queued:
            if max_concurrent > 0 and running_total >= max_concurrent:
                break
            need = task["num_gpus"]
            thr = task["min_free_hbm_gb"]
            thr = cfg["min_free_hbm_gb"] if thr is None else thr
            cands = []
            for gpu in range(ngpu):
                if gpu in reserved:
                    continue
                if counts.get(gpu, 0) >= max_per:
                    continue
                if self._free_hbm_gb(gpu, sample) < thr:
                    continue
                cands.append(gpu)
            if len(cands) < need:
                continue
            # prefer GPUs with fewest running tasks
            cands.sort(key=lambda g: (counts.get(g, 0), g))
            chosen = cands[:need]
            self._launch(task, chosen)
            for g in chosen:
                counts[g] = counts.get(g, 0) + 1
            running_total += 1
            changed = True
            if cooldown > 0:
                # one launch per cooldown window
                break
        return changed

    def _run_logs(self, tid: int) -> list[int]:
        """Run numbers that have a log file for this task, ascending. Run 0 is
        the legacy single combined log (task_<id>.log), if it still exists."""
        runs: list[int] = []
        if (LOG_DIR / f"task_{tid}.log").exists():
            runs.append(0)
        for p in LOG_DIR.glob(f"task_{tid}.*.log"):
            m = re.fullmatch(rf"task_{tid}\.(\d+)\.log", p.name)
            if m:
                runs.append(int(m.group(1)))
        return sorted(set(runs))

    def read_log(self, tid: int, tail: int = 200,
                 run: Optional[int] = None) -> dict:
        """Tail one run's log. ``run`` defaults to the latest. Returns the text
        plus the resolved path, run number and the full list of runs."""
        runs = self._run_logs(tid)
        if run is None:
            run = runs[-1] if runs else None
        if run is None:
            return {"log": "", "path": "", "run": None, "runs": runs}
        fname = f"task_{tid}.log" if run == 0 else f"task_{tid}.{run}.log"
        path = LOG_DIR / fname
        text = ""
        if path.exists():
            try:
                lines = path.read_bytes().decode(errors="replace").splitlines()
                text = "\n".join(lines[-tail:])
            except Exception:
                text = ""
        return {"log": text, "path": str(path), "run": run, "runs": runs}

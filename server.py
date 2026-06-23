"""FastAPI server: live GPU telemetry over WebSocket + task-queue REST API."""
from __future__ import annotations

import asyncio
import collections
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from monitor import GpuMonitor
from scheduler import Scheduler

ROOT = Path(__file__).resolve().parent
HISTORY_LEN = 600          # ~10 min at 1 Hz
SAMPLE_INTERVAL = 1.0
DISPATCH_INTERVAL = 2.0

app = FastAPI(title="gpu-orchestrator")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
monitor = GpuMonitor()
scheduler = Scheduler(monitor)

# rolling per-GPU history: gpu_index -> deque[[ts, util, mem_used, tx, rx]]
history: dict[int, collections.deque] = {
    i: collections.deque(maxlen=HISTORY_LEN) for i in range(monitor.count)
}
_latest: dict = {"ts": 0, "ok": monitor.ok, "gpus": []}
_clients: set[WebSocket] = set()
# bandwidth diffing: only resend the (heavy) task list / config when the
# scheduler's revision changes, and only resend a process' static metadata
# (command line, name, user) the first time we see its pid.
_last_rev = -1
_seen_pids: set[int] = set()
_META_KEYS = ("cmd", "pname", "user")


def _thin_sample(sample: dict) -> dict:
    """Copy of ``sample`` with per-process metadata stripped for pids the
    clients already know about (they cache it). Keeps live numeric fields."""
    gpus_out = []
    cur: set[int] = set()
    for g in sample.get("gpus", []):
        procs_out = []
        for pr in g.get("procs", []):
            cur.add(pr["pid"])
            if pr["pid"] in _seen_pids:
                procs_out.append({k: v for k, v in pr.items()
                                  if k not in _META_KEYS})
            else:
                procs_out.append(pr)
        gpus_out.append({**g, "procs": procs_out})
    _seen_pids.clear()
    _seen_pids.update(cur)
    return {**sample, "gpus": gpus_out}


def _record(sample: dict) -> None:
    global _latest
    _latest = sample
    for g in sample.get("gpus", []):
        history[g["index"]].append([
            round(sample["ts"], 1),
            g.get("util"), g.get("mem_used"),
            g.get("pcie_tx"), g.get("pcie_rx"),
        ])


async def _monitor_loop() -> None:
    global _last_rev
    loop = asyncio.get_event_loop()
    while True:
        sample = await loop.run_in_executor(None, monitor.sample)
        _record(sample)
        await loop.run_in_executor(None, scheduler.reap)
        if not _clients:
            await asyncio.sleep(SAMPLE_INTERVAL)
            continue
        usage = await loop.run_in_executor(None, scheduler.task_usage, sample)
        msg = {"type": "tick", "sample": _thin_sample(sample), "usage": usage,
               "dispatch": scheduler.dispatch_state}
        # the task list is heavy and rarely changes: only resend on bump.
        # (config is intentionally NOT pushed here — it would clobber an input
        #  the user is mid-editing; it ships via snapshot and API responses.)
        rev = scheduler.get_rev()
        if rev != _last_rev:
            msg["tasks"] = scheduler.list_tasks()
            _last_rev = rev
        dead = []
        for ws in list(_clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _clients.discard(ws)
        await asyncio.sleep(SAMPLE_INTERVAL)


def _util_avg(window_s: float = 300.0) -> dict[int, float]:
    """Per-GPU mean utilisation over the last ``window_s`` seconds, from the
    rolling history. Feeds the low-util dispatch gate (max_gpu_util_pct)."""
    now = time.time()
    out: dict[int, float] = {}
    for i, h in history.items():
        vals = [r[1] for r in h if r[1] is not None and now - r[0] <= window_s]
        if vals:
            out[i] = sum(vals) / len(vals)
    return out


async def _dispatch_loop() -> None:
    loop = asyncio.get_event_loop()
    while True:
        await loop.run_in_executor(None, scheduler.tick, _latest, _util_avg())
        await asyncio.sleep(DISPATCH_INTERVAL)


@app.on_event("startup")
async def _startup() -> None:
    asyncio.create_task(_monitor_loop())
    asyncio.create_task(_dispatch_loop())


@app.get("/")
async def index() -> HTMLResponse:
    return HTMLResponse((ROOT / "static" / "index.html").read_text())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _clients.add(ws)
    try:
        await ws.send_json({
            "type": "snapshot",
            "static": monitor.static,
            "history": {i: list(h) for i, h in history.items()},
            "sample": _latest,
            "tasks": scheduler.list_tasks(),
            "config": scheduler.get_config(),
            "dispatch": scheduler.dispatch_state,
            "monitor_ok": monitor.ok,
            "monitor_err": monitor.err,
        })
        while True:
            await ws.receive_text()  # keepalive; ignore content
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


# ---- REST API ---------------------------------------------------------------
class TaskIn(BaseModel):
    command: str
    name: str = ""
    priority: int = 0
    num_gpus: int = 1
    min_free_hbm_gb: float | None = None


class IdsIn(BaseModel):
    ids: list[int]


class UpdateIn(BaseModel):
    ids: list[int]
    priority: int | None = None


class ConfigIn(BaseModel):
    max_tasks_per_gpu: int | None = None
    min_free_hbm_gb: float | None = None
    min_free_ram_gb: float | None = None
    max_concurrent_tasks: int | None = None
    dispatch_cooldown_s: float | None = None
    max_gpu_util_pct: float | None = None
    reserved_gpus: list[int] | None = None
    paused: bool | None = None


class EvacIn(BaseModel):
    gpu: int
    reserve: bool = True
    kill: bool = True


class RunNowIn(BaseModel):
    id: int


class StatusIn(BaseModel):
    ids: list[int]
    status: str


@app.post("/api/tasks")
async def add_task(t: TaskIn):
    tid = scheduler.add_task(t.command, t.name, t.priority, t.num_gpus,
                             t.min_free_hbm_gb)
    return {"id": tid, "tasks": scheduler.list_tasks()}


@app.get("/api/tasks")
async def list_tasks_api():
    return {"tasks": scheduler.list_tasks()}


@app.get("/api/config")
async def get_config_api():
    return {"config": scheduler.get_config()}


@app.get("/api/snapshot")
async def snapshot_api():
    return {"sample": _latest, "tasks": scheduler.list_tasks(),
            "config": scheduler.get_config()}


@app.delete("/api/tasks")
async def del_tasks(b: IdsIn):
    scheduler.delete_tasks(b.ids)
    return {"tasks": scheduler.list_tasks()}


@app.post("/api/tasks/requeue")
async def requeue(b: IdsIn):
    scheduler.requeue_tasks(b.ids)
    return {"tasks": scheduler.list_tasks()}


@app.post("/api/tasks/retry_failed")
async def retry_failed():
    scheduler.retry_failed()
    return {"tasks": scheduler.list_tasks()}


@app.post("/api/tasks/run_now")
async def run_now(b: RunNowIn):
    scheduler.run_now(b.id, _latest)
    return {"tasks": scheduler.list_tasks()}


@app.post("/api/tasks/start")
async def start_tasks(b: IdsIn):
    """一键启动: force-launch selected queued tasks, staggered one per tick."""
    scheduler.run_now_many(b.ids)
    return {"tasks": scheduler.list_tasks()}


@app.post("/api/tasks/pin")
async def pin_tasks(b: IdsIn):
    """置顶: bump selected queued tasks to the top of the queue."""
    scheduler.pin_tasks(b.ids)
    return {"tasks": scheduler.list_tasks()}


@app.post("/api/tasks/set_status")
async def set_status(b: StatusIn):
    scheduler.set_status(b.ids, b.status)
    return {"tasks": scheduler.list_tasks()}


@app.post("/api/tasks/update")
async def update(b: UpdateIn):
    fields = {}
    if b.priority is not None:
        fields["priority"] = b.priority
    scheduler.update_tasks(b.ids, **fields)
    return {"tasks": scheduler.list_tasks()}


@app.get("/api/tasks/{tid}/log")
async def get_log(tid: int, tail: int = 300, run: int | None = None):
    return scheduler.read_log(tid, tail, run)


@app.post("/api/config")
async def set_config(b: ConfigIn):
    kw = {k: v for k, v in b.dict().items() if v is not None}
    return {"config": scheduler.set_config(**kw)}


@app.post("/api/evacuate")
async def evacuate(b: EvacIn):
    scheduler.evacuate_gpu(b.gpu, b.reserve, b.kill)
    return {"tasks": scheduler.list_tasks(),
            "config": scheduler.get_config()}

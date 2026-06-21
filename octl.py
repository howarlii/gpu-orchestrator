#!/usr/bin/env python3
"""octl — command-line client for gpu-orchestrator.

Submit / inspect / control tasks without the web UI. Talks to the server's
REST API (default http://127.0.0.1:8800, override with OCTL_URL).

Examples:
  octl add "cd ~/proj && python train.py --lr 1e-4" -n train-a -p 5
  octl ls
  octl status
  octl rm 3 4 5
  octl requeue 7
  octl prio 10 7 8
  octl config --max-per-gpu 2 --min-hbm 20
  octl config --pause          # / --resume
  octl evac 0                  # kill+requeue tasks on GPU0 and reserve it
  octl log 7
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("OCTL_URL", "http://127.0.0.1:8800").rstrip("/")


def req(path, method="GET", body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return json.load(resp)
    except urllib.error.URLError as e:
        sys.exit(f"octl: cannot reach orchestrator at {BASE}: {e}")


def gb(b):
    return "-" if b is None else f"{b/1024**3:.1f}G"


def gbps(b):
    return "-" if b is None else f"{b/1e9:.1f}G"


def cmd_add(a):
    body = {"command": a.command, "name": a.name or "",
            "priority": a.priority, "num_gpus": a.gpus,
            "min_free_hbm_gb": a.min_hbm}
    d = req("/api/tasks", "POST", body)
    print(f"queued #{d['id']}  {a.name or a.command[:60]}")


def _print_tasks(tasks):
    if not tasks:
        print("(no tasks)")
        return
    print(f"{'ID':>4} {'STATUS':<8} {'PRIO':>4} {'GPU':<6} NAME")
    for t in tasks:
        gpus = ",".join(map(str, json.loads(t["gpu_ids"] or "[]")))
        print(f"{t['id']:>4} {t['status']:<8} {t['priority']:>4} "
              f"{gpus:<6} {t['name'] or ''}")


def cmd_ls(a):
    _print_tasks(req("/api/tasks")["tasks"])


def cmd_status(a):
    d = req("/api/snapshot")
    counts = {}
    for t in d["tasks"]:
        if t["status"] == "running":
            for g in json.loads(t["gpu_ids"] or "[]"):
                counts[g] = counts.get(g, 0) + 1
    cfg = d["config"]
    print(f"config: max/gpu={cfg['max_tasks_per_gpu']} "
          f"min_free_hbm={cfg['min_free_hbm_gb']}G "
          f"reserved={cfg['reserved_gpus']} "
          f"paused={cfg['paused']}")
    print(f"{'GPU':<4} {'UTIL':>5} {'MEM':>14} {'PCIe tx/rx':>16} {'TASKS':>6}")
    for g in d["sample"].get("gpus", []):
        mem = f"{gb(g.get('mem_used'))}/{gb(g.get('mem_total'))}"
        pcie = f"{gbps(g.get('pcie_tx'))}/{gbps(g.get('pcie_rx'))}/s"
        util = "-" if g.get("util") is None else f"{g['util']}%"
        print(f"#{g['index']:<3} {util:>5} {mem:>14} {pcie:>16} "
              f"{counts.get(g['index'], 0):>6}")
    print()
    _print_tasks(d["tasks"])


def cmd_rm(a):
    req("/api/tasks", "DELETE", {"ids": a.ids})
    print(f"deleted {a.ids}")


def cmd_requeue(a):
    req("/api/tasks/requeue", "POST", {"ids": a.ids})
    print(f"requeued {a.ids}")


def cmd_prio(a):
    req("/api/tasks/update", "POST", {"ids": a.ids, "priority": a.priority})
    print(f"set priority {a.priority} on {a.ids}")


def cmd_config(a):
    kw = {}
    if a.max_per_gpu is not None:
        kw["max_tasks_per_gpu"] = a.max_per_gpu
    if a.min_hbm is not None:
        kw["min_free_hbm_gb"] = a.min_hbm
    if a.reserve is not None:
        kw["reserved_gpus"] = a.reserve
    if a.pause:
        kw["paused"] = True
    if a.resume:
        kw["paused"] = False
    cfg = (req("/api/config", "POST", kw)["config"] if kw
           else req("/api/config")["config"])
    print(json.dumps(cfg, indent=2))


def cmd_evac(a):
    req("/api/evacuate", "POST", {"gpu": a.gpu, "reserve": not a.no_reserve})
    print(f"evacuated GPU {a.gpu}" + ("" if a.no_reserve else " (reserved)"))


def cmd_log(a):
    d = req(f"/api/tasks/{a.id}/log?tail={a.tail}")
    print(d.get("log", "") or "(empty)")


def main():
    p = argparse.ArgumentParser(prog="octl")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("add", help="enqueue a shell command")
    s.add_argument("command")
    s.add_argument("-n", "--name", default="")
    s.add_argument("-p", "--priority", type=int, default=0)
    s.add_argument("-g", "--gpus", type=int, default=1, help="num GPUs")
    s.add_argument("--min-hbm", type=float, default=None, dest="min_hbm")
    s.set_defaults(fn=cmd_add)

    sub.add_parser("ls", help="list tasks").set_defaults(fn=cmd_ls)
    sub.add_parser("status", help="GPUs + tasks").set_defaults(fn=cmd_status)

    s = sub.add_parser("rm", help="delete tasks")
    s.add_argument("ids", type=int, nargs="+")
    s.set_defaults(fn=cmd_rm)

    s = sub.add_parser("requeue", help="kill+requeue tasks")
    s.add_argument("ids", type=int, nargs="+")
    s.set_defaults(fn=cmd_requeue)

    s = sub.add_parser("prio", help="set priority on tasks")
    s.add_argument("priority", type=int)
    s.add_argument("ids", type=int, nargs="+")
    s.set_defaults(fn=cmd_prio)

    s = sub.add_parser("config", help="show / set scheduler config")
    s.add_argument("--max-per-gpu", type=int, default=None, dest="max_per_gpu")
    s.add_argument("--min-hbm", type=float, default=None, dest="min_hbm")
    s.add_argument("--reserve", type=int, nargs="*", default=None,
                   help="set reserved GPU list (empty = none)")
    s.add_argument("--pause", action="store_true")
    s.add_argument("--resume", action="store_true")
    s.set_defaults(fn=cmd_config)

    s = sub.add_parser("evac", help="evacuate a GPU")
    s.add_argument("gpu", type=int)
    s.add_argument("--no-reserve", action="store_true")
    s.set_defaults(fn=cmd_evac)

    s = sub.add_parser("log", help="tail a task log")
    s.add_argument("id", type=int)
    s.add_argument("--tail", type=int, default=300)
    s.set_defaults(fn=cmd_log)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

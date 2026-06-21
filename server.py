"""FastAPI server: live GPU telemetry over WebSocket + task-queue REST API."""
from __future__ import annotations

import asyncio
import collections
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
    loop = asyncio.get_event_loop()
    while True:
        sample = await loop.run_in_executor(None, monitor.sample)
        _record(sample)
        await loop.run_in_executor(None, scheduler.reap)
        dead = []
        for ws in list(_clients):
            try:
                await ws.send_json({"type": "tick", "sample": sample,
                                    "tasks": scheduler.list_tasks()})
            except Exception:
                dead.append(ws)
        for ws in dead:
            _clients.discard(ws)
        await asyncio.sleep(SAMPLE_INTERVAL)


async def _dispatch_loop() -> None:
    loop = asyncio.get_event_loop()
    while True:
        await loop.run_in_executor(None, scheduler.tick, _latest)
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
    reserved_gpus: list[int] | None = None
    paused: bool | None = None


class EvacIn(BaseModel):
    gpu: int
    reserve: bool = True


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
async def get_log(tid: int, tail: int = 300):
    return {"log": scheduler.read_log(tid, tail)}


@app.post("/api/config")
async def set_config(b: ConfigIn):
    kw = {k: v for k, v in b.dict().items() if v is not None}
    return {"config": scheduler.set_config(**kw)}


@app.post("/api/evacuate")
async def evacuate(b: EvacIn):
    scheduler.evacuate_gpu(b.gpu, b.reserve)
    return {"tasks": scheduler.list_tasks(),
            "config": scheduler.get_config()}

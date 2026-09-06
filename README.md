# GPU Orchestrator

Single-node multi-GPU monitoring panel + task scheduler. nvitop-style live
view (util / HBM / PCIe history charts, per-process SM/mem/CPU/RAM) plus a
task queue that dispatches single shell commands onto free GPUs under
live-tunable constraints.

## Setup

```bash
cd ~/gpu-orchestrator
./setup.sh          # venv + deps + vendored frontend libs (uPlot, Alpine)
./run.sh            # serves on 0.0.0.0:8800  (PORT=xxxx to override)
```

Open `http://localhost:8800`. If on a remote box, forward the port:
`ssh -L 8800:localhost:8800 <host>`.

## Concepts

- **Task** = one shell command. Launched with `CUDA_VISIBLE_DEVICES` set to its assigned GPU(s), cwd `~`, stdout+stderr → `logs/task_<id>.log`. The original automatic single-/multi-GPU placement remains available. A task can instead name several candidate physical GPUs and run once on whichever candidate becomes available first.
- **Dispatch loop** (every 2s) assigns `queued` tasks to GPUs that satisfy:
  not reserved, running-count `< max_tasks_per_gpu`, free-HBM `≥ min_free_hbm_gb`.
  Higher `priority` dispatched first. Per-task `min_free_hbm_gb` overrides global.
- **Estimated DRAM** is an optional per-task safety gate. Before any normal or forced launch, the scheduler compares `estimated_dram_gb` with the host's current available RAM. The task stays queued when its estimate is larger; tasks without an estimate keep the existing behavior.
- **Reserve / evacuate**: `evac` on a GPU card kills + requeues its tasks and
  marks the GPU reserved (no new dispatch). `free` un-reserves it. To run only
  priority work on a card: evac the others, raise that task's priority.
- **Persistence**: SQLite `orchestrator.db`. On restart, tasks that were
  `running` become `lost` (their processes are not re-adopted) and can be
  requeued.

## Panel

- GPU cards: util%/mem% bars + two history charts (util%/mem%, PCIe tx/rx MB/s)
  + per-process table. PCIe is **device-level only** — NVML does not expose
  per-process PCIe throughput.
- Queue: multi-select, batch delete/requeue/set-priority, click column headers
  to sort (id/name/command-lexicographic/status/priority/gpu/runtime), live log
  viewer with follow.
- Scheduler bar: edit `max_tasks_per_gpu`, `min_free_hbm_gb`, pause dispatch —
  all applied live.
- Candidate GPU/resource binding: fill `candidate GPUs` with values such as `0,2`; the task remains one single-GPU run and the scheduler chooses one available candidate. The command and common args are shared. In `per-GPU args`, optionally map each GPU to the external-resource args that must accompany it, one `GPU=ARGS` per line. For example, GPU 0 can append the path for its paired NVMe while GPU 2 appends a different path. A queued choice is shown as `→0|2`; after launch, `0 ← 0|2` shows both the selected GPU and original candidates. External-resource concurrency follows the existing scheduler settings such as `max_tasks_per_gpu`; the mapping itself adds no extra lock.
- The new-task form and queue table both show candidate/selected GPU information and estimated DRAM. Hover the GPU cell to inspect its per-GPU argument mapping. Hover anywhere on a task row to see the complete raw task record, including the unmodified original command, plus the derived effective command.

The CLI exposes the same setting:

```bash
./octl add "python train.py --epochs 10" --gpu-ids 0,2 --dram 120
./octl add "python train.py --epochs 10" --gpu-ids 0,2 --gpu-arg '0=--data /disk0' --gpu-arg '2=--data /disk2' --dram 120
```

## Files

| file | role |
|------|------|
| `monitor.py`   | NVML + psutil sampling |
| `scheduler.py` | SQLite queue, subprocess launch/kill, dispatch policy |
| `server.py`    | FastAPI: WS telemetry + REST task API |
| `static/index.html` | Alpine + uPlot single-page UI |

# Changelog

## [0.5.0] - 2026-06-24
### Features
- **Occupy GPU (社交预留)**: each GPU card gets an `occupy` button. Clicking it
  prompts for a user name (pre-filled with the server's OS user) and launches a
  tiny placeholder process pinned to that GPU whose command line reads
  `Please reserve this GPU for <user>`. The process holds only a bare CUDA
  context (~550 MB HBM, 0 % util) — just enough to appear in other people's
  `nvidia-smi` so they don't grab the card. A `release` button stops it, and a
  `🔒 <user>` badge shows on occupied cards
  (`POST /api/occupy` / `POST /api/release` → `Scheduler.occupy_gpu` /
  `release_gpu`; placeholder source in `occupy_worker.py`).
### Design Rationale
- The reservation message must show up in **both** `nvidia-smi` and our own
  process table, both of which read `/proc/<pid>/cmdline`. Rather than depend on
  `setproctitle` (not installed; a C extension), the worker is launched via
  `bash -c 'exec -a "<message>" python3'` with its source piped on **stdin**, so
  the process' only argv element — hence its whole cmdline — is exactly the
  message. `CUDA_DEVICE_ORDER=PCI_BUS_ID` pins it to the clicked NVML index;
  `PYTHONHOME=<base_prefix>` silences the fake-argv0 stdlib-path warning.
- The CUDA context is created with raw `ctypes` against `libcuda.so.1` — no
  torch/pycuda dependency — keeping HBM/util footprint minimal.
- Occupy state is persisted in a dedicated `occupy` table and the placeholders
  are launched detached (`setsid`), so they survive a server restart and are
  re-adopted by PID exactly like running tasks.
### Notes & Caveats
- `occupy` is intentionally **decoupled** from the scheduler's `reserved_gpus`
  gate: it is a *social* signal to other humans, not an orchestrator constraint.
  To also stop the orchestrator itself from dispatching onto the card, use the
  existing `evac` / reserve controls.
- The placeholder costs a few hundred MB of HBM (the minimum for a CUDA
  context); it cannot be literally zero.

## [0.4.0] - 2026-06-23
### Features
- **Batch one-click start (一键启动)**: with tasks selected on the *active* tab, a
  `▶ start` button force-launches the selected queued tasks. As an explicit
  operator override it **bypasses** the RAM gate, dispatch cooldown, pause, HBM
  gate and per-GPU cap; only the one-launch-per-tick spacing (~2 s) and
  reserved-GPU avoidance still apply, so the batch starts promptly but is still
  staggered rather than firing literally all at once
  (`POST /api/tasks/start` → `Scheduler.run_now_many`).
- **Pin to top (置顶)**: `set prio` (numeric) is replaced by a `📌 pin top`
  button that floats selected queued tasks above the rest — "run first" with no
  explicit priority value (`POST /api/tasks/pin` → `Scheduler.pin_tasks`).
- **Selection-aware toolbar**: `start` / `pin` / `requeue` / `delete` are hidden
  when nothing is selected.
- **Dispatch delay applies to every launch**: the dispatcher now launches **at
  most one task per tick** (forced launches included), giving each freshly
  started task time to claim RAM/HBM before the next dispatch decision —
  directly fixing the "multiple tasks start at once → RAM blow-up" issue.
- **Low-util dispatch gate**: new `max_gpu_util_pct` config — a GPU whose recent
  **5-min average utilization** is at/above this value is skipped for new
  dispatch, preventing a busy/full GPU from being over-stuffed with tasks
  (0 = disabled). The 5-min average is computed server-side from the rolling
  history and passed into `Scheduler.tick`.
- **Dispatch status display**: the Scheduler panel shows a live `dispatch` line
  explaining the current state — `dispatching` / `idle` / `paused` / `cooldown`
  / `blocked` — and, when blocked, *why* (RAM gate, cooldown, max-concurrent, or
  a per-GPU breakdown of reserved / at-cap / busy / low-HBM). Pushed every tick
  as `dispatch` (`Scheduler.dispatch_state`).

### Design Rationale
- Force-started tasks deliberately bypass every dispatch gate (RAM, cooldown,
  pause, HBM, per-GPU cap, concurrency) — it is an explicit manual override, so
  it trusts the operator. The one-launch-per-tick spacing is retained as the
  only safety against an instantaneous RAM spike from a large batch.
- The util gate is a *limit*, not an override: it reads "only keep dispatching to
  a GPU while its 5-min avg util is below the threshold", so an already-saturated
  GPU stops receiving new work even if it has spare HBM and task-cap headroom.
- One-launch-per-tick replaces the old "one-per-cooldown-window only when
  cooldown > 0" rule, so even with no cooldown configured, launches are spaced by
  the 2 s dispatch interval instead of bursting within a single tick.

### Notes & Caveats
- A force-started task that needs more GPUs than are currently unreserved will
  block forced dispatch and surface that as the `dispatch` reason rather than
  silently downgrading; free a GPU or delete the task to unblock.
- Server **not** restarted for this change set (per request) — the new behavior
  takes effect on the next restart.

## [0.3.0] - 2026-06-22
### Features
- **Per-task resource tooltip**: hovering a `running` status pill shows that
  task's live GPU(s), HBM, SM%, CPU% and RAM, aggregated over the task's whole
  process tree (`scheduler.task_usage()`, pushed each tick as `usage`).
- **Pending tasks in the process table**: tasks that are `running` but have not
  yet allocated any GPU memory now appear in the Processes table with a
  `starting` tag, so a just-launched task is visible before it claims HBM.
- **Dispatch cooldown** *(optional)*: new `dispatch_cooldown_s` config — after a
  launch, the dispatcher holds off for N seconds (launches one task per window)
  so a freshly-started task can actually claim HBM before the next is assigned,
  preventing burst over-allocation (0 = disabled).
- **GPU evac choice**: the per-GPU `evac` button now offers two actions —
  *kill all* (kill+requeue everything on it, then reserve) or *no new* (reserve
  only / drain — leave running tasks alone). Backed by `evacuate_gpu(kill=…)`.
- **Run now**: queued tasks get a `run now` action that force-launches them
  immediately on the GPU(s) with the most free HBM, bypassing the per-GPU cap,
  HBM gate and cooldown (`POST /api/tasks/run_now`).
- **Bandwidth reduction**: the heavy task list is only re-broadcast when the
  scheduler state actually changes (revision counter), and per-process metadata
  (command line, name, user) is sent only the first time a pid appears — clients
  cache it. Live numeric telemetry still streams at 1 Hz.

### Design Rationale
- The task list (with full commands) and per-process command lines were re-sent
  to every client every second; both are near-static, so a `rev` gate plus a
  client-side pid→metadata cache cut steady-state WebSocket traffic sharply
  without changing the 1 Hz chart cadence.
- Config is deliberately *not* pushed on ticks — it would overwrite an input the
  user is mid-editing; it ships via the snapshot and API responses instead.
- `run now` bypasses the gates on purpose: it is an explicit manual override, so
  it trusts the operator rather than the scheduler's safety thresholds.

### Notes & Caveats
- The `pause dispatch` checkbox and the redundant `reserved: …` label were
  removed from the scheduler controls (reserved GPUs are already shown on the
  cards); the `paused` config still exists in the backend for the CLI.
- The bulk `set prio` now applies on **Enter** (the `apply` button is gone), and
  `set prio` / bulk `requeue` are hidden on the *finished* tab. Per-row `requeue`
  (↻) remains for finished/running tasks.

## [0.2.0] - 2026-06-22
### Features
- **Host panel (CPU/RAM/disk/net)**: compact single-row card below the GPU grid —
  CPU total util bar (0–100%), RAM bar with used/total (GiB), disk R/W and net U/D
  rates. Sampled via `psutil` in `monitor.sample_system()` and attached to every
  telemetry sample as `sample.sys`.
- **Scheduler RAM gate**: new `min_free_ram_gb` config — the dispatcher only
  launches new tasks while host free RAM stays above this value (0 = disabled).
- **Global concurrency cap**: new `max_concurrent_tasks` config — total running
  tasks never exceed this threshold (0 = unlimited).
- **One-click retry**: "↻ retry all failed" button on the *finished* tab requeues
  every `failed`/`lost` task at once (`POST /api/tasks/retry_failed`).
- **Crash-resilient queue**: on restart, tasks that were `running` are now
  *re-adopted* by PID instead of being blindly marked `lost`. Children are
  launched with `setsid()`, so they survive an unexpected exit of the Python
  process; the scheduler polls re-adopted PIDs and reaps them when they finish.

### Design Rationale
- Task-queue state has always lived in SQLite (`orchestrator.db`), so the queue
  already survived restarts; the only gap was live processes being orphaned.
  Re-adoption closes that gap without a heavier supervisor/pidfile mechanism.
- RAM gate is a per-tick global check (matches "只有空闲 RAM 高于阈值才继续分配");
  combined with `max_concurrent_tasks` it bounds how aggressively a single tick
  can launch work.

### Notes & Caveats
- Re-adopted (orphaned) processes are not our children, so we cannot `waitpid()`
  them — we poll PID liveness. When such a task ends we mark it `done` with no
  exit code, since the real exit status is unobtainable. PID-reuse is guarded by
  comparing the process' `create_time()` against the task's `started_at`.

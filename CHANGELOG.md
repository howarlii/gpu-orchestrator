# Changelog

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

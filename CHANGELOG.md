# Changelog

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

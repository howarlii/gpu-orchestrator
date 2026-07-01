"""GPU + per-process sampling via NVML and psutil.

Device-level: util / HBM / PCIe Rx-Tx / temp / power.
Per-process:  SM% / GPU-mem / CPU% / RAM  (PCIe is NOT available per-process,
NVML only exposes it at the device level).
"""
from __future__ import annotations

import time
from typing import Optional

import psutil

try:
    import warnings
    warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")
    import pynvml
    _NVML_OK = True
except Exception:  # pragma: no cover
    _NVML_OK = False


def _decode(b) -> str:
    if isinstance(b, bytes):
        return b.decode(errors="replace")
    return str(b)


class GpuMonitor:
    def __init__(self) -> None:
        self.ok = False
        self.err: Optional[str] = None
        self.count = 0
        self.handles: list = []
        self.static: list[dict] = []
        # per-pid caches
        self._proc_cache: dict[int, psutil.Process] = {}
        self._last_seen_ts: dict[int, int] = {}
        # system-wide rate counters (disk / net are cumulative -> need deltas)
        self._prev_io: Optional[dict] = None
        try:
            psutil.cpu_percent(None)  # prime; first call returns 0.0
        except Exception:
            pass
        self._init()

    def _init(self) -> None:
        if not _NVML_OK:
            self.err = "pynvml not importable"
            return
        try:
            pynvml.nvmlInit()
            self.count = pynvml.nvmlDeviceGetCount()
            for i in range(self.count):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.handles.append(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                try:
                    plimit = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                except Exception:
                    plimit = None
                self.static.append({
                    "index": i,
                    "name": _decode(pynvml.nvmlDeviceGetName(h)),
                    "mem_total": mem.total,
                    "power_limit": plimit,
                })
            self.ok = True
        except Exception as e:  # pragma: no cover
            self.err = f"{type(e).__name__}: {e}"

    # ---- per-process helpers ------------------------------------------------
    def _get_proc(self, pid: int) -> Optional[psutil.Process]:
        p = self._proc_cache.get(pid)
        if p is not None and p.is_running():
            return p
        try:
            p = psutil.Process(pid)
            p.cpu_percent(None)  # prime; first call returns 0.0
            self._proc_cache[pid] = p
            return p
        except Exception:
            return None

    def _cpu_ram(self, pid: int) -> tuple[float, int]:
        """CPU% (since last sample) and RSS bytes, aggregated over children."""
        p = self._get_proc(pid)
        if p is None:
            return 0.0, 0
        cpu = 0.0
        rss = 0
        try:
            cpu += p.cpu_percent(None)
            rss += p.memory_info().rss
            for c in p.children(recursive=True):
                cc = self._get_proc(c.pid)
                if cc is None:
                    continue
                try:
                    cpu += cc.cpu_percent(None)
                    rss += cc.memory_info().rss
                except Exception:
                    pass
        except Exception:
            pass
        return round(cpu, 1), rss

    def _proc_util(self, h, idx: int) -> dict[int, dict]:
        """pid -> {sm, mem} utilisation since last query for this device."""
        out: dict[int, dict] = {}
        last = self._last_seen_ts.get(idx, 0)
        try:
            samples = pynvml.nvmlDeviceGetProcessUtilization(h, last)
            for s in samples:
                out[s.pid] = {"sm": s.smUtil, "mem": s.memUtil}
                self._last_seen_ts[idx] = max(self._last_seen_ts.get(idx, 0),
                                              s.timeStamp)
        except Exception:
            pass
        return out

    # ---- system-wide (CPU / RAM / disk / net) -------------------------------
    def sample_system(self, ts: float) -> dict:
        """Cheap host-level stats: CPU%, RAM, disk R/W and net U/D rates."""
        out: dict = {}
        try:
            out["cpu"] = psutil.cpu_percent(None)
        except Exception:
            out["cpu"] = None
        try:
            vm = psutil.virtual_memory()
            out["ram_used"] = vm.total - vm.available
            out["ram_total"] = vm.total
        except Exception:
            out["ram_used"] = out["ram_total"] = None
        try:
            d = psutil.disk_io_counters()
            # Sum only real (non-loopback) NICs. The default aggregate counter
            # includes `lo`, where a local proxy's relayed traffic is counted on
            # BOTH sent and recv (and again on the physical NIC), inflating the
            # host's apparent throughput several-fold. Excluding loopback gives
            # the true external network rate.
            per = psutil.net_io_counters(pernic=True)
            ns = sum(s.bytes_sent for name, s in per.items()
                     if not name.startswith("lo"))
            nr = sum(s.bytes_recv for name, s in per.items()
                     if not name.startswith("lo"))
            prev = self._prev_io
            dt = (ts - prev["ts"]) if prev else 0.0
            if prev and dt > 0:
                out["disk_r"] = max(0.0, (d.read_bytes - prev["dr"]) / dt)
                out["disk_w"] = max(0.0, (d.write_bytes - prev["dw"]) / dt)
                out["net_u"] = max(0.0, (ns - prev["ns"]) / dt)
                out["net_d"] = max(0.0, (nr - prev["nr"]) / dt)
            else:
                out["disk_r"] = out["disk_w"] = out["net_u"] = out["net_d"] = 0.0
            self._prev_io = {"ts": ts, "dr": d.read_bytes, "dw": d.write_bytes,
                             "ns": ns, "nr": nr}
        except Exception:
            out["disk_r"] = out["disk_w"] = out["net_u"] = out["net_d"] = None
        return out

    # ---- main sample --------------------------------------------------------
    def sample(self) -> dict:
        ts = time.time()
        if not self.ok:
            return {"ts": ts, "ok": False, "err": self.err, "gpus": [],
                    "sys": self.sample_system(ts)}
        gpus = []
        for i, h in enumerate(self.handles):
            st = self.static[i]
            g = {
                "index": i,
                "name": st["name"],
                "mem_total": st["mem_total"],
                "power_limit": st["power_limit"],
                "procs": [],
            }
            try:
                u = pynvml.nvmlDeviceGetUtilizationRates(h)
                g["util"] = u.gpu
                g["mem_util"] = u.memory
            except Exception:
                g["util"] = None
                g["mem_util"] = None
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                g["mem_used"] = mem.used
            except Exception:
                g["mem_used"] = None
            try:
                g["temp"] = pynvml.nvmlDeviceGetTemperature(
                    h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                g["temp"] = None
            try:
                g["power"] = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                g["power"] = None
            # PCIe throughput is reported in KB/s -> convert to bytes/s
            try:
                g["pcie_tx"] = pynvml.nvmlDeviceGetPcieThroughput(
                    h, pynvml.NVML_PCIE_UTIL_TX_BYTES) * 1024
                g["pcie_rx"] = pynvml.nvmlDeviceGetPcieThroughput(
                    h, pynvml.NVML_PCIE_UTIL_RX_BYTES) * 1024
            except Exception:
                g["pcie_tx"] = None
                g["pcie_rx"] = None

            # processes
            putil = self._proc_util(h, i)
            procs: dict[int, dict] = {}
            for fn in (pynvml.nvmlDeviceGetComputeRunningProcesses,
                       pynvml.nvmlDeviceGetGraphicsRunningProcesses):
                try:
                    for pr in fn(h):
                        used = getattr(pr, "usedGpuMemory", None)
                        if used in (None, 2**64 - 1):
                            used = 0
                        procs.setdefault(pr.pid, {"pid": pr.pid,
                                                  "gpu_mem": 0})
                        procs[pr.pid]["gpu_mem"] += int(used or 0)
                except Exception:
                    pass
            for pid, pr in procs.items():
                cpu, rss = self._cpu_ram(pid)
                util = putil.get(pid, {})
                name = ""
                cmd = ""
                user = ""
                try:
                    pp = self._get_proc(pid)
                    if pp:
                        name = pp.name()
                        cmd = " ".join(pp.cmdline())
                        try:
                            user = pp.username()
                        except Exception:
                            user = ""
                except Exception:
                    pass
                pr.update({
                    "sm": util.get("sm"),
                    "cpu": cpu,
                    "rss": rss,
                    "pname": name,
                    "user": user,
                    "cmd": cmd,
                })
            g["procs"] = sorted(procs.values(),
                                key=lambda x: -x.get("gpu_mem", 0))
            gpus.append(g)
        return {"ts": ts, "ok": True, "gpus": gpus,
                "sys": self.sample_system(ts)}

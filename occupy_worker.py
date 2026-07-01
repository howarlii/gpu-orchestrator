"""Minimal GPU placeholder ("occupy") process.

Creates a bare CUDA context on the single visible GPU (CUDA_VISIBLE_DEVICES)
and then sleeps forever. The context costs a few hundred MB of HBM and 0%
compute — just enough to appear in ``nvidia-smi``'s process list so other
users see the GPU is spoken for.

This script takes NO arguments and reads nothing: it is launched via
``exec -a "<message>" python3`` with its source piped on stdin, so the
process' argv[0] (hence /proc/<pid>/cmdline, which both nvidia-smi and our
own monitor read) is exactly the reservation message.
"""
import ctypes
import time


def main() -> None:
    cuda = ctypes.CDLL("libcuda.so.1")
    if cuda.cuInit(0) != 0:
        raise SystemExit("cuInit failed")
    dev = ctypes.c_int()
    if cuda.cuDeviceGet(ctypes.byref(dev), 0) != 0:
        raise SystemExit("cuDeviceGet failed")
    ctx = ctypes.c_void_p()
    # cuCtxCreate(&ctx, flags=0, dev) — a primary-ish context, minimal HBM.
    if cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) != 0:
        raise SystemExit("cuCtxCreate failed")
    # default SIGTERM terminates us; the scheduler kills the process group
    # (and escalates to SIGKILL) on release.
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()

from types import SimpleNamespace
from unittest import TestCase, mock

from monitor import GpuMonitor, _parse_gds_totals


class GdsStatsTest(TestCase):
    def test_parses_driver_totals(self):
        text = """
Reads : n=4 ok=4 err=0 readMiB=12.5 io_state_err=0
Writes : n=2 ok=2 err=0 writeMiB=3 io_state_err=0
"""
        self.assertEqual(_parse_gds_totals(text), (12.5 * 1024**2, 3 * 1024**2))

    def test_missing_counters_returns_none(self):
        self.assertIsNone(_parse_gds_totals("IO stats: Disabled"))


class SystemSamplingTest(TestCase):
    def test_uses_physical_disks_and_diffs_gds_counters(self):
        monitor = GpuMonitor.__new__(GpuMonitor)
        monitor._prev_io = None
        monitor._prev_gds = None
        monitor._disk_devices = {"nvme0n1"}
        monitor._gds_totals = mock.Mock(side_effect=[
            ("enabled", (10 * 1024**2, 20 * 1024**2)),
            ("enabled", (11 * 1024**2, 22 * 1024**2)),
        ])
        memory = SimpleNamespace(
            total=1000, available=400, free=200, cached=250, buffers=50,
            slab=25,
        )
        first = {
            "nvme0n1": SimpleNamespace(
                read_bytes=100, write_bytes=200, busy_time=1000),
            "nvme0n1p1": SimpleNamespace(
                read_bytes=10000, write_bytes=20000, busy_time=10000),
        }
        second = {
            "nvme0n1": SimpleNamespace(
                read_bytes=300, write_bytes=500, busy_time=1500),
            "nvme0n1p1": SimpleNamespace(
                read_bytes=50000, write_bytes=60000, busy_time=20000),
        }
        network = {"eth0": SimpleNamespace(bytes_sent=100, bytes_recv=200)}

        with mock.patch("monitor.psutil.cpu_percent", return_value=10), \
             mock.patch("monitor.psutil.virtual_memory", return_value=memory), \
             mock.patch("monitor.psutil.disk_io_counters",
                        side_effect=[first, second]), \
             mock.patch("monitor.psutil.net_io_counters", return_value=network):
            monitor.sample_system(1.0)
            result = monitor.sample_system(2.0)

        self.assertEqual(result["disk_devices"], ["nvme0n1"])
        self.assertEqual(result["disk_r"], 200)
        self.assertEqual(result["disk_w"], 300)
        self.assertEqual(result["disk_busy"], 50)
        self.assertEqual(result["gds_r"], 1024**2)
        self.assertEqual(result["gds_w"], 2 * 1024**2)

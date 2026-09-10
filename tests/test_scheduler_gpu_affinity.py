import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scheduler


GIB = 1024**3


class FakeMonitor:
    count = 4


def sample(free_gb=None):
    free_gb = free_gb or {}
    return {
        "gpus": [
            {
                "index": gpu,
                "mem_total": 80 * GIB,
                "mem_used": (80 - free_gb.get(gpu, 70)) * GIB,
            }
            for gpu in range(4)
        ]
    }


class GpuAffinityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.patches = [
            mock.patch.object(scheduler, "DB_PATH", root / "test.db"),
            mock.patch.object(scheduler, "LOG_DIR", root / "logs"),
        ]
        for patch in self.patches:
            patch.start()
        scheduler.LOG_DIR.mkdir()
        self.scheduler = scheduler.Scheduler(FakeMonitor())
        self.scheduler.set_config(max_concurrent_tasks=8)
        self.launches = []

        def launch(task, gpu_ids):
            self.launches.append(
                (task["id"], gpu_ids,
                 self.scheduler._task_command(task, gpu_ids))
            )
            self.scheduler.db.execute(
                "UPDATE tasks SET status='running', gpu_ids=? WHERE id=?",
                (str(gpu_ids), task["id"]),
            )
            self.scheduler.db.commit()

        self.scheduler._launch = launch

    def tearDown(self):
        self.scheduler.db.close()
        for patch in reversed(self.patches):
            patch.stop()
        self.tmp.cleanup()

    def test_candidate_task_runs_once_with_selected_gpu_args(self):
        tid = self.scheduler.add_task(
            "python train.py --epochs 10",
            target_gpu_ids=[2, 0],
            gpu_args={2: "--data /disk2", 0: "--data /disk0"},
        )

        tasks = self.scheduler.list_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["id"], tid)
        self.assertEqual(tasks[0]["num_gpus"], 1)
        self.assertEqual(json.loads(tasks[0]["target_gpu_ids"]), [2, 0])
        self.assertEqual(
            json.loads(tasks[0]["gpu_args"]),
            {"2": "--data /disk2", "0": "--data /disk0"},
        )

        self.scheduler.tick(sample({0: 5}))
        self.assertEqual(
            self.launches,
            [(tid, [2], "python train.py --epochs 10 --data /disk2")],
        )

    def test_candidates_do_not_fall_back_to_an_unlisted_gpu(self):
        tid = self.scheduler.add_task("true", target_gpu_ids=[1])

        self.scheduler.tick(sample({1: 5}))

        self.assertEqual(self.launches, [])
        reason = self.scheduler.dispatch_state["reason"]
        self.assertIn("candidate GPU(s) [1]", reason)
        self.assertIn(f"task #{tid}", reason)

    def test_run_now_chooses_candidate_and_uses_its_args(self):
        tid = self.scheduler.add_task(
            "run", target_gpu_ids=[1, 3],
            gpu_args={1: "--disk one", 3: "--disk three"},
        )

        launched = self.scheduler.run_now(tid, sample({1: 1, 3: 5}))

        self.assertTrue(launched)
        self.assertEqual(self.launches, [(tid, [3], "run --disk three")])

    def test_force_start_stays_within_candidates(self):
        tid = self.scheduler.add_task("true", target_gpu_ids=[3, 1])
        self.scheduler.run_now_many([tid])

        self.scheduler.tick(sample({3: 60, 1: 20}))

        self.assertEqual(self.launches, [(tid, [3], "true")])

    def test_start_selected_only_requeues_paused_task(self):
        tid = self.scheduler.add_task("true", target_gpu_ids=[2])
        self.scheduler.pause_tasks([tid])
        self.scheduler.set_config(paused=True)

        self.scheduler.run_now_many([tid])

        task = self.scheduler.list_tasks()[0]
        self.assertEqual(task["status"], "queued")
        self.assertNotIn(tid, self.scheduler._force_ids)

        self.scheduler.tick(sample())

        self.assertEqual(self.launches, [])
        self.assertEqual(self.scheduler.list_tasks()[0]["status"], "queued")

    def test_concurrent_tasks_bind_each_gpu_to_its_resource_args(self):
        first = self.scheduler.add_task(
            "run", target_gpu_ids=[0, 1],
            gpu_args={0: "--disk zero", 1: "--disk one"},
        )
        second = self.scheduler.add_task(
            "run", target_gpu_ids=[0, 1],
            gpu_args={0: "--disk zero", 1: "--disk one"},
        )
        self.scheduler.tick(sample())
        self.scheduler.tick(sample())

        self.assertEqual(
            self.launches,
            [
                (first, [0], "run --disk zero"),
                (second, [1], "run --disk one"),
            ],
        )

    def test_rejects_duplicate_or_out_of_range_targets(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.scheduler.add_task("true", target_gpu_ids=[1, 1])
        with self.assertRaisesRegex(ValueError, "out of range"):
            self.scheduler.add_task("true", target_gpu_ids=[4])
        with self.assertRaisesRegex(ValueError, "unselected"):
            self.scheduler.add_task(
                "true", target_gpu_ids=[1], gpu_args={2: "--x"}
            )

    def test_dram_estimate_blocks_task_but_allows_smaller_queued_task(self):
        self.scheduler.set_config(min_free_ram_gb=20)
        large = self.scheduler.add_task("large", estimated_dram_gb=50)
        small = self.scheduler.add_task("small", estimated_dram_gb=40)

        vm = mock.Mock(available=64 * GIB)
        with mock.patch.object(scheduler.psutil, "virtual_memory",
                               return_value=vm):
            self.scheduler.tick(sample())

        self.assertEqual(self.launches, [(small, [0], "small")])
        tasks = {task["id"]: task for task in self.scheduler.list_tasks()}
        self.assertEqual(tasks[large]["status"], "queued")
        self.assertEqual(tasks[large]["estimated_dram_gb"], 50)

    def test_dram_estimate_explains_why_task_is_waiting(self):
        self.scheduler.set_config(min_free_ram_gb=20)
        tid = self.scheduler.add_task("large", estimated_dram_gb=50)

        vm = mock.Mock(available=64 * GIB)
        with mock.patch.object(scheduler.psutil, "virtual_memory",
                               return_value=vm):
            self.scheduler.tick(sample())

        self.assertEqual(self.launches, [])
        self.assertEqual(
            self.scheduler.dispatch_state["reason"],
            f"task #{tid} needs 70.0G free RAM: estimated DRAM 50.0G + "
            "min free RAM 20.0G > available 64.0G",
        )

    def test_dram_estimate_also_guards_manual_start_paths(self):
        self.scheduler.set_config(min_free_ram_gb=20)
        run_now = self.scheduler.add_task("run-now", estimated_dram_gb=50)
        forced = self.scheduler.add_task("forced", estimated_dram_gb=50)
        vm = mock.Mock(available=64 * GIB)

        with mock.patch.object(scheduler.psutil, "virtual_memory",
                               return_value=vm):
            self.assertFalse(self.scheduler.run_now(run_now, sample()))
            self.scheduler.run_now_many([forced])
            self.scheduler.tick(sample())

        self.assertEqual(self.launches, [])
        self.assertIn("estimated DRAM 50.0G + min free RAM 20.0G",
                      self.scheduler.dispatch_state["reason"])

    def test_combined_dram_gate_launches_at_exact_required_memory(self):
        self.scheduler.set_config(min_free_ram_gb=24)
        tid = self.scheduler.add_task("fits", estimated_dram_gb=40)
        vm = mock.Mock(available=64 * GIB)

        with mock.patch.object(scheduler.psutil, "virtual_memory",
                               return_value=vm):
            self.scheduler.tick(sample())

        self.assertEqual(self.launches, [(tid, [0], "fits")])

    def test_run_now_honors_min_free_ram_without_dram_estimate(self):
        self.scheduler.set_config(min_free_ram_gb=70)
        tid = self.scheduler.add_task("run-now")
        vm = mock.Mock(available=64 * GIB)

        with mock.patch.object(scheduler.psutil, "virtual_memory",
                               return_value=vm):
            self.assertFalse(self.scheduler.run_now(tid, sample()))

        self.assertIn("estimated DRAM 0.0G + min free RAM 70.0G",
                      self.scheduler.dispatch_state["reason"])

    def test_rejects_invalid_dram_estimate(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.scheduler.add_task("true", estimated_dram_gb=-1)

    def test_edit_task_updates_launch_settings(self):
        tid = self.scheduler.add_task("old", num_gpus=2)

        updated = self.scheduler.edit_task(
            tid,
            "python train.py",
            name="edited",
            priority=7,
            num_gpus=3,
            min_free_hbm_gb=18,
            target_gpu_ids=[3, 1],
            gpu_args={3: "--disk three", 1: "--disk one"},
            estimated_dram_gb=96.5,
        )

        self.assertTrue(updated)
        task = self.scheduler.list_tasks()[0]
        self.assertEqual(task["name"], "edited")
        self.assertEqual(task["command"], "python train.py")
        self.assertEqual(task["priority"], 7)
        self.assertEqual(task["num_gpus"], 1)
        self.assertEqual(json.loads(task["target_gpu_ids"]), [3, 1])
        self.assertEqual(
            json.loads(task["gpu_args"]),
            {"3": "--disk three", "1": "--disk one"},
        )
        self.assertEqual(task["min_free_hbm_gb"], 18)
        self.assertEqual(task["estimated_dram_gb"], 96.5)

    def test_edit_task_can_restore_automatic_gpu_placement(self):
        tid = self.scheduler.add_task(
            "run", target_gpu_ids=[0, 2], gpu_args={0: "--disk zero"}
        )

        self.scheduler.edit_task(
            tid, "run", num_gpus=2, target_gpu_ids=None, gpu_args=None
        )

        task = self.scheduler.list_tasks()[0]
        self.assertEqual(task["num_gpus"], 2)
        self.assertEqual(task["target_gpu_ids"], "")
        self.assertEqual(task["gpu_args"], "")

    def test_edit_task_rejects_running_task_without_changing_it(self):
        tid = self.scheduler.add_task(
            "old", target_gpu_ids=[0], estimated_dram_gb=32
        )
        self.scheduler.db.execute(
            "UPDATE tasks SET status='running', gpu_ids='[0]' WHERE id=?",
            (tid,),
        )
        self.scheduler.db.commit()

        with self.assertRaisesRegex(ValueError, "pause a running task"):
            self.scheduler.edit_task(
                tid, "new", target_gpu_ids=[1], estimated_dram_gb=64
            )

        task = self.scheduler.list_tasks()[0]
        self.assertEqual(task["command"], "old")
        self.assertEqual(json.loads(task["target_gpu_ids"]), [0])
        self.assertEqual(task["estimated_dram_gb"], 32)


if __name__ == "__main__":
    unittest.main()

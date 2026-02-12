import json
import os
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock dataio modules before importing services
sys.modules["dataio"] = MagicMock()
sys.modules["dataio.data_factory"] = MagicMock()
sys.modules["dataio.data_loader"] = MagicMock()

from services import dosing_scheduler


class DummyThread:
    def __init__(self, target=None, daemon=None):
        self._target = target
        self._daemon = daemon
        self._alive = False

    def start(self):
        self._alive = True

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self._alive = False


class TestDosingScheduler(unittest.TestCase):
    def setUp(self):
        dosing_scheduler.app.testing = True
        self.client = dosing_scheduler.app.test_client()
        self._reset_state()

    def tearDown(self):
        dosing_scheduler.stop_scheduler()
        self._reset_state()

    def _reset_state(self):
        with dosing_scheduler.state_lock:
            dosing_scheduler.scheduler_running = False
            dosing_scheduler.scheduler_thread = None
        with dosing_scheduler.result_lock:
            dosing_scheduler.latest_result = None
        dosing_scheduler.schedule.clear(dosing_scheduler.SCHEDULER_TAG)

    def test_refresh_scheduler_settings_valid_and_invalid(self):
        valid_cfg = {
            "scheduler": {
                "enabled": True,
                "auto_start": False,
                "frequency": {
                    "type": "hourly",
                    "interval_hours": 2,
                    "minute": 10,
                },
            }
        }
        with patch("services.dosing_scheduler.load_config", return_value=valid_cfg):
            settings = dosing_scheduler._refresh_scheduler_settings()
        self.assertEqual(settings["frequency"]["interval_hours"], 2)
        self.assertEqual(settings["frequency"]["minute"], 10)
        self.assertFalse(settings["auto_start"])

        invalid_cfg = {
            "scheduler": {
                "enabled": True,
                "frequency": {
                    "type": "daily",
                    "interval_hours": 0,
                    "minute": 99,
                },
            }
        }
        with patch("services.dosing_scheduler.load_config", return_value=invalid_cfg):
            settings = dosing_scheduler._refresh_scheduler_settings()
        self.assertEqual(settings["frequency"]["type"], "hourly")
        self.assertEqual(settings["frequency"]["interval_hours"], 1)
        self.assertEqual(settings["frequency"]["minute"], 5)

    def test_scheduler_lifecycle_start_stop_idempotent(self):
        settings = {
            "enabled": True,
            "auto_start": True,
            "frequency": {"type": "hourly", "interval_hours": 1, "minute": 5},
        }
        with patch("services.dosing_scheduler._refresh_scheduler_settings", return_value=settings), patch(
            "services.dosing_scheduler.threading.Thread", DummyThread
        ):
            ok, reason = dosing_scheduler.start_scheduler()
            self.assertTrue(ok)
            self.assertEqual(reason, "started")

            ok, reason = dosing_scheduler.start_scheduler()
            self.assertFalse(ok)
            self.assertEqual(reason, "already_running")

            ok, reason = dosing_scheduler.stop_scheduler()
            self.assertTrue(ok)
            self.assertEqual(reason, "stopped")

            ok, reason = dosing_scheduler.stop_scheduler()
            self.assertTrue(ok)
            self.assertEqual(reason, "already_stopped")

    def test_scheduled_job_success(self):
        mock_data_dict = {"pool_1": np.random.rand(60, 6).astype(np.float32)}
        mock_last_dt = datetime(2026, 2, 11, 12, 0, 0)
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"pool_1": {"status": "success"}}
        mock_pipeline.predictor_manager.config = {"seq_len": 60, "features": ["a"]}

        with patch("services.dosing_scheduler.read_data", return_value=(mock_data_dict, mock_last_dt)), patch(
            "services.dosing_scheduler.get_pipeline", return_value=mock_pipeline
        ):
            dosing_scheduler.scheduled_optimization_job()

        with dosing_scheduler.result_lock:
            result = dosing_scheduler.latest_result
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "success")
        self.assertTrue(result["upload_skipped"])
        self.assertEqual(result["pool_count"], 1)
        self.assertIn("result", result)

    def test_scheduled_job_failure(self):
        mock_pipeline = MagicMock()
        mock_pipeline.predictor_manager.config = {"seq_len": 60, "features": ["a"]}
        with patch("services.dosing_scheduler.read_data", side_effect=ValueError("boom")), patch(
            "services.dosing_scheduler.get_pipeline", return_value=mock_pipeline
        ):
            dosing_scheduler.scheduled_optimization_job()

        with dosing_scheduler.result_lock:
            result = dosing_scheduler.latest_result
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["upload_skipped"])
        self.assertIn("error", result)
        self.assertIn("traceback", result)

    def test_scheduler_management_apis(self):
        settings = {
            "enabled": True,
            "auto_start": True,
            "frequency": {"type": "hourly", "interval_hours": 1, "minute": 5},
        }
        with patch("services.dosing_scheduler._refresh_scheduler_settings", return_value=settings), patch(
            "services.dosing_scheduler.threading.Thread", DummyThread
        ):
            r1 = self.client.post("/alum_dosing/scheduler/start")
            r2 = self.client.post("/alum_dosing/scheduler/start")
            r3 = self.client.post("/alum_dosing/scheduler/stop")
            r4 = self.client.post("/alum_dosing/scheduler/stop")

        d1 = json.loads(r1.data)
        d2 = json.loads(r2.data)
        d3 = json.loads(r3.data)
        d4 = json.loads(r4.data)

        self.assertEqual(d1["status"], "success")
        self.assertEqual(d2["status"], "already_running")
        self.assertEqual(d3["status"], "success")
        self.assertEqual(d4["status"], "success")

        status_resp = self.client.get("/alum_dosing/scheduler/status")
        status_data = json.loads(status_resp.data)
        self.assertIn("scheduler_running", status_data)
        self.assertIn("has_latest_result", status_data)
        self.assertIn("next_run_at", status_data)
        self.assertIn("timestamp", status_data)

        latest_resp = self.client.get("/alum_dosing/latest_result")
        latest_data = json.loads(latest_resp.data)
        self.assertEqual(latest_data["status"], "no_result")

        sample_latest = {
            "status": "success",
            "executed_at": "2026-02-12 12:00:00",
            "duration_ms": 12,
            "upload_skipped": True,
            "pool_count": 1,
            "result": {"pool_1": {"status": "success"}},
        }
        with dosing_scheduler.result_lock:
            dosing_scheduler.latest_result = sample_latest

        latest_resp = self.client.get("/alum_dosing/latest_result")
        latest_data = json.loads(latest_resp.data)
        self.assertEqual(latest_data["status"], "success")
        self.assertIn("result", latest_data)
        self.assertEqual(latest_data["result"]["upload_skipped"], True)


if __name__ == "__main__":
    unittest.main()

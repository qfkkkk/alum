import json
import os
import sys
import unittest


# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services import dosing_scheduler
from services.dosing_api import get_pipeline


class TestDosingSchedulerIntegration(unittest.TestCase):
    def setUp(self):
        dosing_scheduler.app.testing = True
        self.client = dosing_scheduler.app.test_client()
        self._reset_state()

        pipeline = get_pipeline()
        self.enabled_pools = pipeline.predictor_manager.enabled_pools or ["pool_1"]
        cfg = pipeline.predictor_manager.config
        self.pred_len = int(cfg.get("pred_len", 6))
        self.control_horizon = int(cfg.get("control_horizon", 5))

    def tearDown(self):
        dosing_scheduler.stop_scheduler()
        self._reset_state()

    def _reset_state(self):
        with dosing_scheduler.state_lock:
            dosing_scheduler.scheduler_running = False
            dosing_scheduler.scheduler_thread = None
        with dosing_scheduler.result_lock:
            dosing_scheduler.latest_predict_result = None
            dosing_scheduler.latest_optimize_result = None
        dosing_scheduler.schedule.clear(dosing_scheduler.SCHEDULER_TAG_PREDICT)
        dosing_scheduler.schedule.clear(dosing_scheduler.SCHEDULER_TAG_OPTIMIZE)

    def test_predict_job_integration(self):
        dosing_scheduler.scheduled_predict_job()

        with dosing_scheduler.result_lock:
            result = dosing_scheduler.latest_predict_result

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["task"], "predict")
        self.assertEqual(result["pool_count"], len(self.enabled_pools))
        self.assertIn("result", result)

        for pool_name in self.enabled_pools:
            self.assertIn(pool_name, result["result"])
            self.assertEqual(len(result["result"][pool_name]), self.pred_len)

    def test_optimize_job_integration(self):
        dosing_scheduler.scheduled_optimization_job()

        with dosing_scheduler.result_lock:
            result = dosing_scheduler.latest_optimize_result

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["task"], "optimize")
        self.assertEqual(result["pool_count"], len(self.enabled_pools))
        self.assertIn("result", result)

        for pool_name in self.enabled_pools:
            self.assertIn(pool_name, result["result"])
            pool_res = result["result"][pool_name]
            self.assertIn("recommendations", pool_res)
            self.assertEqual(len(pool_res["recommendations"]), self.control_horizon)
            self.assertNotIn("predictions", pool_res)

    def test_scheduler_lifecycle_and_status_api(self):
        ok, reason = dosing_scheduler.start_scheduler()
        self.assertTrue(ok)
        self.assertEqual(reason, "started")

        status_resp = self.client.get("/alum_dosing/scheduler/status")
        status_data = json.loads(status_resp.data)
        self.assertTrue(status_data["success"])
        self.assertEqual(status_data["code"], "OK")
        scheduler_data = status_data["data"]["scheduler"]
        self.assertTrue(scheduler_data["running"])
        self.assertIn("predict", scheduler_data["tasks"])
        self.assertIn("optimize", scheduler_data["tasks"])

        dosing_scheduler.scheduled_predict_job()
        dosing_scheduler.scheduled_optimization_job()

        latest_resp = self.client.get("/alum_dosing/latest_result")
        latest_data = json.loads(latest_resp.data)
        self.assertTrue(latest_data["success"])
        self.assertEqual(latest_data["code"], "OK")
        self.assertIn("predict", latest_data["data"]["latest"])
        self.assertIn("optimize", latest_data["data"]["latest"])

        ok, reason = dosing_scheduler.stop_scheduler()
        self.assertTrue(ok)
        self.assertIn(reason, ("stopped", "already_stopped"))


if __name__ == "__main__":
    unittest.main()

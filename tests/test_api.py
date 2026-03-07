import unittest
import json
import sys
import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.dosing_api import app

class TestDosingAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_check(self):
        response = self.app.get('/alum_dosing/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['code'], 'OK')
        self.assertEqual(data['message'], 'healthy')
        self.assertEqual(data['data']['service'], 'alum_dosing')
        self.assertEqual(data['data']['health'], 'healthy')
        self.assertIn('timestamp', data['meta'])

    def _assert_predict_response(self, response):
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['code'], 'OK')
        self.assertEqual(data['data']['task'], 'predict')
        self.assertIn('pools', data['data'])
        self.assertIn('point_count', data['data'])
        self.assertIn('executed_at', data['data'])
        self.assertIn('timestamp', data['meta'])

        pools = data['data']['pools']
        if pools:
            first_pool = pools[0]
            self.assertIn('pool_id', first_pool)
            self.assertIn('forecast', first_pool)
            if first_pool['forecast']:
                self.assertIn('datetime', first_pool['forecast'][0])
                self.assertIn('turbidity_pred', first_pool['forecast'][0])

    def _assert_optimize_response(self, response):
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['code'], 'OK')
        self.assertEqual(data['data']['task'], 'optimize')
        self.assertIn('pools', data['data'])
        self.assertIn('executed_at', data['data'])
        self.assertIn('timestamp', data['meta'])

        pools = data['data']['pools']
        if pools:
            first_pool = pools[0]
            self.assertIn('pool_id', first_pool)
            self.assertIn('recommendations', first_pool)
            self.assertNotIn('predictions', first_pool)
            self.assertNotIn('turbidity_predictions', first_pool)
            if first_pool['recommendations']:
                rec = first_pool['recommendations'][0]
                self.assertIn('datetime', rec)
                self.assertIn('value', rec)

    def _build_full_fake_input_data(self):
        pools = {}
        for pool_idx in range(1, 5):
            rows = []
            for t in range(60):
                row = []
                for f in range(6):
                    row.append(round(pool_idx + t * 0.01 + f * 0.1, 4))
                rows.append(row)
            pools[f"pool_{pool_idx}"] = rows
        return pools

    def _build_mock_predict_context(self):
        config = {
            "seq_len": 4,
            "time_interval_minutes": 5,
            "features": ["dose", "turb_chushui", "turb_jinshui", "flow", "pH", "temp_shuimian"],
        }
        baseline = {
            "pool_1": np.zeros((4, 6), dtype=np.float32),
            "pool_2": np.ones((4, 6), dtype=np.float32),
        }
        last_dt = datetime(2026, 2, 13, 12, 15, 0)

        class DummyPipeline:
            def __init__(self, cfg):
                self.predictor_manager = SimpleNamespace(config=cfg)
                self.captured_data = None
                self.captured_last_dt = None

            def predict_only(self, input_data, input_last_dt):
                self.captured_data = input_data
                self.captured_last_dt = input_last_dt
                return {"pool_1": {"2026-02-13 12:20:00": 1.2345}}

        return DummyPipeline(config), baseline, last_dt

    def _build_full_fake_predictions(self):
        base_dt = datetime(2026, 2, 13, 12, 0, 0)
        preds = {}
        for pool_idx in range(1, 5):
            pool_name = f"pool_{pool_idx}"
            series = {}
            for step in range(1, 7):
                dt = base_dt + timedelta(minutes=5 * step)
                series[dt.strftime("%Y-%m-%d %H:%M:%S")] = round(1.0 + pool_idx * 0.1 + step * 0.02, 4)
            preds[pool_name] = series
        return preds

    def _build_full_fake_features(self):
        return {
            "pool_1": {"current_dose": 10.0, "ph": 7.10, "flow": 1200.0},
            "pool_2": {"current_dose": 11.0, "ph": 7.05, "flow": 1180.0},
            "pool_3": {"current_dose": 9.8, "ph": 7.20, "flow": 1210.0},
            "pool_4": {"current_dose": 10.5, "ph": 7.15, "flow": 1195.0},
        }

    def test_predict_get_not_allowed(self):
        response = self.app.get('/alum_dosing/predict')
        self.assertEqual(response.status_code, 405)

    def test_predict_post_missing_mode_bad_request(self):
        response = self.app.post(
            '/alum_dosing/predict',
            data=json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data["success"])
        self.assertEqual(data["code"], "PREDICT_API_BAD_REQUEST")

    def test_predict_post_external_patch_data(self):
        pipeline, baseline, last_dt = self._build_mock_predict_context()
        payload = {
            "mode": "external",
            "patches": {
                "pool_1": [
                    {
                        "datetime": "2026-02-13 12:05:00",
                        "features": {"dose": 9.9, "pH": 7.2},
                    },
                    {
                        "datetime": "2026-02-13 12:15:00",
                        "features": {"flow": 123.0},
                    },
                ]
            },
        }
        with (
            patch("services.dosing_api.get_pipeline", return_value=pipeline),
            patch("services.dosing_api.read_data", return_value=(baseline, last_dt)),
        ):
            response = self.app.post(
                "/alum_dosing/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )

        self._assert_predict_response(response)
        self.assertAlmostEqual(float(pipeline.captured_data["pool_1"][1, 0]), 9.9, places=6)
        self.assertAlmostEqual(float(pipeline.captured_data["pool_1"][1, 4]), 7.2, places=6)
        self.assertAlmostEqual(float(pipeline.captured_data["pool_1"][3, 3]), 123.0, places=6)
        self.assertTrue(np.allclose(pipeline.captured_data["pool_2"], np.ones((4, 6), dtype=np.float32)))
        self.assertEqual(pipeline.captured_last_dt, last_dt)

    def test_predict_post_online_mode(self):
        payload = {"mode": "online"}
        response = self.app.post(
            '/alum_dosing/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self._assert_predict_response(response)

    def test_predict_post_agent_patch_data(self):
        pipeline, baseline, last_dt = self._build_mock_predict_context()
        payload = {
            "mode": "agent",
            "patches": {
                "pool_1": [
                    {
                        "datetime": "2026-02-13 12:10:00",
                        "features": {"ph": 6.8},
                    }
                ]
            },
        }
        with (
            patch("services.dosing_api.get_pipeline", return_value=pipeline),
            patch("services.dosing_api.read_data", return_value=(baseline, last_dt)),
        ):
            response = self.app.post(
                "/alum_dosing/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )

        self._assert_predict_response(response)
        self.assertAlmostEqual(float(pipeline.captured_data["pool_1"][2, 4]), 6.8, places=6)

    def test_predict_post_external_missing_patches_bad_request(self):
        pipeline, baseline, last_dt = self._build_mock_predict_context()
        payload = {"mode": "external"}
        with (
            patch("services.dosing_api.get_pipeline", return_value=pipeline),
            patch("services.dosing_api.read_data", return_value=(baseline, last_dt)),
        ):
            response = self.app.post(
                "/alum_dosing/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data["success"])
        self.assertEqual(data["code"], "PREDICT_API_BAD_REQUEST")

    def test_predict_post_external_patch_datetime_out_of_window(self):
        pipeline, baseline, last_dt = self._build_mock_predict_context()
        payload = {
            "mode": "external",
            "patches": {
                "pool_1": [
                    {
                        "datetime": "2026-02-13 12:30:00",
                        "features": {"dose": 8.8},
                    }
                ]
            },
        }
        with (
            patch("services.dosing_api.get_pipeline", return_value=pipeline),
            patch("services.dosing_api.read_data", return_value=(baseline, last_dt)),
        ):
            response = self.app.post(
                "/alum_dosing/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data["success"])
        self.assertEqual(data["code"], "PREDICT_PATCH_DATETIME_OUT_OF_WINDOW")
        self.assertIn("不在当前窗口范围", data["error"]["detail"])

    def test_predict_post_external_patch_datetime_second_mismatch_bad_request(self):
        pipeline, baseline, last_dt = self._build_mock_predict_context()
        payload = {
            "mode": "external",
            "patches": {
                "pool_1": [
                    {
                        "datetime": "2026-02-13 12:10:01",
                        "features": {"dose": 8.8},
                    }
                ]
            },
        }
        with (
            patch("services.dosing_api.get_pipeline", return_value=pipeline),
            patch("services.dosing_api.read_data", return_value=(baseline, last_dt)),
        ):
            response = self.app.post(
                "/alum_dosing/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data["success"])
        self.assertEqual(data["code"], "PREDICT_PATCH_DATETIME_OUT_OF_WINDOW")
        self.assertIn("不在当前窗口范围", data["error"]["detail"])

    def test_predict_post_external_patch_feature_not_found_bad_request(self):
        pipeline, baseline, last_dt = self._build_mock_predict_context()
        payload = {
            "mode": "external",
            "patches": {
                "pool_1": [
                    {
                        "datetime": "2026-02-13 12:10:00",
                        "features": {"unknown_feature": 8.8},
                    }
                ]
            },
        }
        with (
            patch("services.dosing_api.get_pipeline", return_value=pipeline),
            patch("services.dosing_api.read_data", return_value=(baseline, last_dt)),
        ):
            response = self.app.post(
                "/alum_dosing/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data["success"])
        self.assertEqual(data["code"], "PREDICT_PATCH_FEATURE_NOT_FOUND")
        self.assertIn("不存在", data["error"]["detail"])

    def test_predict_post_external_patch_datetime_format_invalid_bad_request(self):
        pipeline, baseline, last_dt = self._build_mock_predict_context()
        payload = {
            "mode": "external",
            "patches": {
                "pool_1": [
                    {
                        "datetime": "2026-02-13 12:10",
                        "features": {"dose": 8.8},
                    }
                ]
            },
        }
        with (
            patch("services.dosing_api.get_pipeline", return_value=pipeline),
            patch("services.dosing_api.read_data", return_value=(baseline, last_dt)),
        ):
            response = self.app.post(
                "/alum_dosing/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data["success"])
        self.assertEqual(data["code"], "PREDICT_PATCH_DATETIME_FORMAT_INVALID")
        self.assertIn("YYYY-mm-dd HH:MM:SS", data["error"]["detail"])

    def test_predict_post_external_patch_aligns_last_dt_to_grid(self):
        pipeline, baseline, _ = self._build_mock_predict_context()
        raw_last_dt = datetime(2026, 2, 13, 12, 15, 42)
        payload = {
            "mode": "external",
            "patches": {
                "pool_1": [
                    {
                        "datetime": "2026-02-13 12:15:00",
                        "features": {"dose": 8.8},
                    }
                ]
            },
        }
        with (
            patch("services.dosing_api.get_pipeline", return_value=pipeline),
            patch("services.dosing_api.read_data", return_value=(baseline, raw_last_dt)),
        ):
            response = self.app.post(
                "/alum_dosing/predict",
                data=json.dumps(payload),
                content_type="application/json",
            )
        self._assert_predict_response(response)
        self.assertEqual(pipeline.captured_last_dt, datetime(2026, 2, 13, 12, 15, 0))
        self.assertAlmostEqual(float(pipeline.captured_data["pool_1"][3, 0]), 8.8, places=6)

    def test_predict_post_bad_mode(self):
        payload = {
            "mode": "bad_mode",
            "data_dict": self._build_full_fake_input_data(),
        }
        response = self.app.post(
            '/alum_dosing/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data["success"])
        self.assertEqual(data["code"], "PREDICT_API_BAD_REQUEST")

    def test_optimize_get_not_allowed(self):
        response = self.app.get('/alum_dosing/optimize')
        self.assertEqual(response.status_code, 405)

    def test_optimize_post_online_mode(self):
        payload = {"mode": "online"}
        response = self.app.post(
            '/alum_dosing/optimize',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self._assert_optimize_response(response)

    def test_optimize_post_with_predictions(self):
        payload = {
            "mode": "external",
            "predictions": self._build_full_fake_predictions(),
            "current_features": self._build_full_fake_features(),
        }
        response = self.app.post(
            '/alum_dosing/optimize',
            data=json.dumps(payload),
            content_type='application/json'
        )
        print('test_optimize_post_with_predictions_response:', response.data.decode('utf-8'))

        self._assert_optimize_response(response)
        data = json.loads(response.data)
        self.assertGreaterEqual(data["data"]["pool_count"], 1)
        self.assertGreaterEqual(data["data"]["point_count"], 1)

    def test_optimize_post_bad_request(self):
        # 缺少 mode，预期 400
        payload = {
            "predictions": {
                "pool_1": {"2026-02-13 12:05:00": 1.11}
            }
        }
        response = self.app.post(
            '/alum_dosing/optimize',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data["success"])
        self.assertEqual(data["code"], "OPTIMIZE_API_BAD_REQUEST")

    def test_optimize_post_external_missing_features_bad_request(self):
        # mode=external 但缺少 current_features，预期 400
        payload = {
            "mode": "external",
            "predictions": {
                "pool_1": {"2026-02-13 12:05:00": 1.11}
            }
        }
        response = self.app.post(
            '/alum_dosing/optimize',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertFalse(data["success"])
        self.assertEqual(data["code"], "OPTIMIZE_API_BAD_REQUEST")

if __name__ == '__main__':
    unittest.main()

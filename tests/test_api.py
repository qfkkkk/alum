import unittest
import json
import sys
import os
from datetime import datetime, timedelta

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

    def test_predict_get(self):
        # 走真实链路：read_data(local) -> predict_only
        response = self.app.get('/alum_dosing/predict')
        self._assert_predict_response(response)

    def test_predict_post_empty_payload_fallback(self):
        # 兼容旧行为：POST 空请求体仍走内部 read_data 链路
        response = self.app.post(
            '/alum_dosing/predict',
            data=json.dumps({}),
            content_type='application/json'
        )
        self._assert_predict_response(response)

    def test_predict_post_external_data(self):
        payload = {
            "last_dt": "2026-02-13 12:00:00",
            "data_dict": self._build_full_fake_input_data()
        }
        response = self.app.post(
            '/alum_dosing/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        print('test_predict_post_external_data_response:', response.data.decode('utf-8'))

        self._assert_predict_response(response)
        data = json.loads(response.data)
        self.assertGreaterEqual(data["data"]["pool_count"], 1)
        self.assertGreaterEqual(data["data"]["point_count"], 1)

    def test_optimize_get(self):
        # 走真实链路：read_data(local) -> pipeline.run
        response = self.app.get('/alum_dosing/optimize')
        self._assert_optimize_response(response)

    def test_optimize_post_empty_payload_fallback(self):
        # 兼容旧行为：POST 空请求体仍走全流程
        response = self.app.post(
            '/alum_dosing/optimize',
            data=json.dumps({}),
            content_type='application/json'
        )
        self._assert_optimize_response(response)

    def test_optimize_post_with_predictions(self):
        payload = {
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
        # 缺少 current_features，预期 400
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

if __name__ == '__main__':
    unittest.main()

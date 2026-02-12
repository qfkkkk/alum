from unittest.mock import MagicMock, patch
import unittest
import json
import sys
import os
from datetime import datetime, timedelta

import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dataio module to avoid config loading issues during test
sys.modules['dataio'] = MagicMock()
sys.modules['dataio.data_factory'] = MagicMock()
sys.modules['dataio.data_loader'] = MagicMock()

from services.dosing_api import app
from utils.config_loader import load_config

class TestDosingAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.last_dt = datetime(2026, 2, 11, 12, 0, 0)

        # 参考 tests/test_pipeline.py 的输入构造方式
        config = load_config()
        self.seq_len = config.get('seq_len', 60)
        self.n_features = len(config.get('features', []))
        self.input_data = np.random.rand(self.seq_len, self.n_features).astype(np.float32)
        pools_cfg = config.get('pools', {})
        self.enabled_pools = [pid for pid, cfg in pools_cfg.items() if cfg.get('enabled', False)] or ['pool_1']

    def _mock_data_read(self):
        data_dict = {
            pool_name: self.input_data
            for pool_name in self.enabled_pools
        }
        return data_dict, self.last_dt

    def test_health_check(self):
        response = self.app.get('/alum_dosing/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['service'], 'alum_dosing')

    def test_predict_turbidity(self):
        # 使用真实 predict_only，只替换输入数据来源
        with patch('services.dosing_api.data_read', side_effect=self._mock_data_read):
            response = self.app.post(
                '/alum_dosing/predict_turbidity',
                data=json.dumps({}),
                content_type='application/json'
            )
        print("======="*10)
        print("predict_turbidity:\n",response.data)
        print("======="*10)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('predictions', data)
        self.assertIn('count', data)
        self.assertIn('timestamp', data)
        print(f"predict_turbidity count={data['count']}")
        
        # Check first prediction structure
        if data['predictions']:
            first_pool = data['predictions'][0]
            self.assertIn('pool_id', first_pool)
            self.assertIn('forecast', first_pool)
            if first_pool['forecast']:
                self.assertIn('datetime', first_pool['forecast'][0])
                self.assertIn('turbidity_pred', first_pool['forecast'][0])
                print(f"first forecast sample={first_pool['forecast'][0]}")

    def test_full_optimization(self):
        # 使用真实 full pipeline，只替换输入数据来源
        with patch('services.dosing_api.data_read', side_effect=self._mock_data_read):
            response = self.app.post(
                '/alum_dosing/full_optimization',
                data=json.dumps({}),
                content_type='application/json'
            )
        print("======="*10)
        print("full_optimization:\n",response.data)
        print("======="*10)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        self.assertIn('results', data)
        self.assertIn('timestamp', data)
        
        # Check result structure
        if data['results']:
            first_pool = data['results'][0]
            self.assertIn('pool_id', first_pool)
            self.assertIn('turbidity_predictions', first_pool)
            self.assertIn('recommendations', first_pool)
            
            # Check recommendations content
            if first_pool['recommendations']:
                rec = first_pool['recommendations'][0]
                self.assertIn('datetime', rec)
                self.assertIn('value', rec)

if __name__ == '__main__':
    unittest.main()

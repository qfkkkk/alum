import unittest
import json
import sys
import os

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

    def test_predict(self):
        # 走真实链路：read_data(local) -> predict_only
        response = self.app.post(
            '/alum_dosing/predict',
            data=json.dumps({}),
            content_type='application/json'
        )
        print('test_predict_response:', response.data)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['code'], 'OK')
        self.assertEqual(data['data']['task'], 'predict')
        self.assertIn('pools', data['data'])
        self.assertIn('point_count', data['data'])
        self.assertIn('executed_at', data['data'])
        self.assertIn('timestamp', data['meta'])
        
        # Check first prediction structure
        pools = data['data']['pools']
        if pools:
            first_pool = pools[0]
            self.assertIn('pool_id', first_pool)
            self.assertIn('forecast', first_pool)
            if first_pool['forecast']:
                self.assertIn('datetime', first_pool['forecast'][0])
                self.assertIn('turbidity_pred', first_pool['forecast'][0])

    def test_optimize(self):
        # 走真实链路：read_data(local) -> pipeline.run
        response = self.app.post(
            '/alum_dosing/optimize',
            data=json.dumps({}),
            content_type='application/json'
        )
        print('test_optimize_response', response.data)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['code'], 'OK')
        self.assertEqual(data['data']['task'], 'optimize')
        self.assertIn('pools', data['data'])
        self.assertIn('executed_at', data['data'])
        self.assertIn('timestamp', data['meta'])
        
        # Check result structure
        pools = data['data']['pools']
        if pools:
            first_pool = pools[0]
            self.assertIn('pool_id', first_pool)
            self.assertIn('recommendations', first_pool)
            self.assertNotIn('predictions', first_pool)
            self.assertNotIn('turbidity_predictions', first_pool)
            
            # Check recommendations content
            if first_pool['recommendations']:
                rec = first_pool['recommendations'][0]
                self.assertIn('datetime', rec)
                self.assertIn('value', rec)

if __name__ == '__main__':
    unittest.main()

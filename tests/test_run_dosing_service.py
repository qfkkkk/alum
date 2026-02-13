import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import unittest
from unittest.mock import MagicMock, patch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services import run_dosing_service


class TestRunDosingService(unittest.TestCase):
    def _is_service_ready(self, timeout: float = 1.0) -> bool:
        try:
            with urllib.request.urlopen("http://localhost:5001/alum_dosing/health", timeout=timeout):
                return True
        except urllib.error.URLError:
            return False

    def test_run_api_only(self):
        with patch("services.run_dosing_service.run_flask_app") as mock_run:
            run_dosing_service.run_api_only()
            mock_run.assert_called_once_with()

    def test_run_scheduler_only(self):
        with (
            patch("services.run_dosing_service.start_scheduler", return_value=(True, "started")) as mock_start,
            patch("services.run_dosing_service.stop_scheduler") as mock_stop,
            patch("services.run_dosing_service.time.sleep", side_effect=KeyboardInterrupt),
        ):
            run_dosing_service.run_scheduler_only()
            mock_start.assert_called_once_with()
            mock_stop.assert_called_once_with()

    def test_show_status(self):
        fake_response = MagicMock()
        fake_response.read.return_value = b'{"success": true}'

        with (
            patch("services.run_dosing_service.urllib.request.urlopen") as mock_urlopen,
            patch("builtins.print") as mock_print,
        ):
            mock_urlopen.return_value.__enter__.return_value = fake_response
            run_dosing_service.show_status()

            mock_urlopen.assert_called_once_with(
                "http://localhost:5001/alum_dosing/scheduler/status", timeout=5
            )
            mock_print.assert_called_once_with('{"success": true}')

    def test_show_status_with_real_service(self):
        if self._is_service_ready():
            self.skipTest("5001 端口已有服务，跳过本地启动集成测试")

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        proc = subprocess.Popen(
            [sys.executable, "-m", "services.run_dosing_service", "api-only"],
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            deadline = time.time() + 20
            while time.time() < deadline:
                if self._is_service_ready():
                    break
                if proc.poll() is not None:
                    self.fail(f"api-only 服务提前退出，returncode={proc.returncode}")
                time.sleep(0.5)
            else:
                self.fail("api-only 服务未在 20 秒内就绪")

            with patch("builtins.print") as mock_print:
                run_dosing_service.show_status()

            self.assertTrue(mock_print.called)
            output = mock_print.call_args[0][0]
            payload = json.loads(output)
            self.assertTrue(payload["success"])
            self.assertEqual(payload["code"], "OK")
            self.assertIn("scheduler", payload["data"])
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait(timeout=5)

    def test_main_dispatch_api_only(self):
        with (
            patch(
                "services.run_dosing_service.argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(mode="api-only"),
            ),
            patch("services.run_dosing_service.run_api_only") as mock_api,
        ):
            run_dosing_service.main()
            mock_api.assert_called_once_with()

    def test_main_dispatch_scheduler_only(self):
        with (
            patch(
                "services.run_dosing_service.argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(mode="scheduler-only"),
            ),
            patch("services.run_dosing_service.run_scheduler_only") as mock_scheduler,
        ):
            run_dosing_service.main()
            mock_scheduler.assert_called_once_with()

    def test_main_dispatch_full(self):
        with (
            patch(
                "services.run_dosing_service.argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(mode="full"),
            ),
            patch("services.run_dosing_service.run_full_service") as mock_full,
        ):
            run_dosing_service.main()
            mock_full.assert_called_once_with()

    def test_main_dispatch_status(self):
        with (
            patch(
                "services.run_dosing_service.argparse.ArgumentParser.parse_args",
                return_value=argparse.Namespace(mode="status"),
            ),
            patch("services.run_dosing_service.show_status") as mock_status,
        ):
            run_dosing_service.main()
            mock_status.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

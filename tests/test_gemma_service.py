import contextlib
import importlib.util
import io
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "llm_gemma_service" / "gemma_service.py"


def load_service(root):
    spec = importlib.util.spec_from_file_location("gemma_service_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = Path(root)
    module.SERVER_LOG = Path(root) / "server.log"
    module.STATE_FILE = Path(root) / "service.json"
    module.LOCK_FILE = Path(root) / "service.lock"
    return module


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.service = load_service(self.directory.name)
        self.state = {"pid": 12345, "identity": {"boot_id": "test", "start_ticks": "17"},
                      "base_url": self.service.BASE, "gpu_verified": False}

    def test_ready_legacy_server_is_reused_across_fresh_imports(self):
        for service in (self.service, load_service(self.directory.name)):
            service.api = Mock(side_effect=lambda path, **kwargs: (
                {"status": "ok"} if path == "/health" else {"data": [{"id": "gemma"}]}))
            service._launch = Mock()
            with contextlib.redirect_stdout(io.StringIO()):
                result = service.ensure_running()
            self.assertTrue(result["reused"])
            service._launch.assert_not_called()
            self.assertFalse(service.STATE_FILE.exists())

    def test_absent_server_starts_once_and_persists_gpu_verification(self):
        service = self.service
        service._ready = Mock(side_effect=[False, False, True])
        service._owned_alive = lambda state: state is not None
        service._launch = Mock(return_value=(self.state, Mock()))
        service.SERVER_LOG.write_text("load_tensors: offloaded 41/41 layers to GPU\n")
        with contextlib.redirect_stdout(io.StringIO()):
            result = service.ensure_running()
        self.assertFalse(result["reused"])
        self.assertEqual(result["gpu_layers"], (41, 41))
        self.assertTrue(service._read_state()["gpu_verified"])
        service._launch.assert_called_once()
        # A fresh import can recover ownership from the file, not notebook globals.
        new_client = load_service(self.directory.name)
        new_client._ready = Mock(return_value=True)
        new_client._owned_alive = lambda state: state is not None
        new_client._launch = Mock()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(new_client.ensure_running()["reused"])
        new_client._launch.assert_not_called()

    def test_loading_process_is_waited_on_without_relaunch(self):
        service = self.service
        service._save_state(self.state)
        service._ready = Mock(return_value=False)
        service._owned_alive = lambda state: state is not None
        service._launch = Mock()
        service._wait_ready = Mock(return_value=0.1)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(service.ensure_running()["reused"])
        service._launch.assert_not_called()
        service._wait_ready.assert_called_once()

    def test_concurrent_callers_do_not_create_two_models(self):
        service = self.service
        ready = threading.Event()
        service._ready = ready.is_set
        service._owned_alive = lambda state: state is not None

        def launch():
            service._save_state(self.state)
            return self.state, Mock()

        def wait(state, timeout, process=None):
            time.sleep(0.05)
            state["gpu_verified"] = True
            service._save_state(state)
            ready.set()
            return 0.05

        service._launch = Mock(side_effect=launch)
        service._wait_ready = wait
        with contextlib.redirect_stdout(io.StringIO()), ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: service.ensure_running(timeout=3), range(2)))
        service._launch.assert_called_once()
        self.assertEqual(sorted(result["reused"] for result in results), [False, True])

    def test_wrong_model_is_not_restarted(self):
        service = self.service
        service.api = Mock(side_effect=lambda path, **kwargs: (
            {"status": "ok"} if path == "/health" else {"data": [{"id": "other"}]}))
        service._launch = Mock()
        with self.assertRaisesRegex(RuntimeError, "другая модель"):
            service.ensure_running()
        service._launch.assert_not_called()

    def test_partial_gpu_load_is_rejected_and_new_launch_cleaned_up(self):
        service = self.service
        service._ready = Mock(side_effect=[False, False, True])
        service._owned_alive = lambda state: state is not None
        service._launch = Mock(return_value=(self.state, Mock()))
        service._stop_owned = Mock()
        service.SERVER_LOG.write_text("offloaded 10/41 layers to GPU")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "10/41"):
            service.ensure_running()
        service._stop_owned.assert_called_once_with(self.state)

    def test_interrupt_keeps_loading_process_for_reconnection(self):
        service = self.service
        service._ready = Mock(return_value=False)
        service._owned_alive = lambda state: state is not None
        service._launch = Mock(return_value=(self.state, Mock()))
        service._wait_ready = Mock(side_effect=KeyboardInterrupt)
        service._stop_owned = Mock()
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            service.ensure_running()
        service._stop_owned.assert_not_called()

    def test_stale_pid_cannot_stop_unrelated_process(self):
        service = self.service
        service._identity = Mock(return_value={"boot_id": "test", "start_ticks": "999"})
        with patch.object(service.os, "killpg") as kill:
            self.assertFalse(service._stop_owned(self.state))
        kill.assert_not_called()

    def test_request_uses_running_server_and_returns_timing(self):
        service = self.service
        service.ensure_running = Mock(return_value={"reused": True})
        service.api = Mock(return_value={"choices": [{"message": {"content": "Ответ"}}],
                                        "usage": {"completion_tokens": 2}})
        with contextlib.redirect_stdout(io.StringIO()):
            result = service.measure_request("Вопрос", max_tokens=12)
        self.assertEqual(result["text"], "Ответ")
        self.assertGreaterEqual(result["seconds"], 0)
        self.assertEqual(service.api.call_args.args[1]["max_tokens"], 12)
        self.assertFalse(service.api.call_args.args[1]["stream"])


if __name__ == "__main__":
    unittest.main()

import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends import BACKENDS  # noqa: E402
from backends import rest_api_backend  # noqa: E402

HEALTH = 'http://gpu-box:9080/running'
MODEL = 'whisper-small-en'


class FakeLocal:
    """Stands in for the local whisper.cpp backend."""

    reinit_on_idle = True
    reinit_on_resume = True

    def __init__(self, manager):
        self.manager = manager
        self._last_use_time = 0.0
        self.current_model = 'base.en'
        self.calls = []

    def initialize(self):
        self.manager.current_model = 'base.en'
        return True

    def transcribe(self, audio_data, sample_rate, language_override=None):
        self.calls.append(len(audio_data))
        return 'local text'

    def reinitialize(self):
        return True

    def cleanup(self):
        pass


def _backend(settings=None, running=None, post=None, health_error=None):
    values = {
        'rest_endpoint_url': 'http://gpu-box:9080/v1/audio/transcriptions',
        'rest_body': {'model': MODEL, 'prompt': ''},
        'rest_fallback_backend': 'vulkan',
        'rest_health_url': HEALTH,
        **(settings or {}),
    }
    config = mock.Mock()
    config.get_setting.side_effect = lambda key, default=None: values.get(key, default)
    manager = SimpleNamespace(
        config=config, current_model=None, ready=False, temp_dir='/tmp', _last_use_time=0.0
    )
    backend = BACKENDS['rest-api'](manager)
    requests = mock.Mock()
    requests.exceptions = rest_api_backend.require_package('requests').exceptions
    if health_error:
        requests.get.side_effect = health_error
    else:
        requests.get.return_value = mock.Mock(
            raise_for_status=mock.Mock(),
            json=mock.Mock(return_value={'running': running or []}),
        )
    requests.post.side_effect = post or (
        lambda *a, **k: mock.Mock(status_code=200, json=mock.Mock(return_value={'text': 'remote text'}))
    )
    backend._requests = requests
    with mock.patch.dict(BACKENDS, {'vulkan': FakeLocal}):
        assert backend.initialize()
    return backend, requests


AUDIO = np.zeros(16000 * 10, dtype=np.float32)
READY = [{'model': MODEL, 'state': 'ready'}]


class RestFallbackRoutingTests(unittest.TestCase):
    def test_remote_when_model_loaded(self):
        backend, requests = _backend(running=READY)
        self.assertEqual(backend.transcribe(AUDIO, 16000), 'remote text')
        self.assertEqual(backend.last_route['served_by'], 'remote')
        self.assertEqual(backend.last_route['model'], MODEL)
        # Deadline scales with length: 2 s + 0.1 s/s x 10 s.
        self.assertAlmostEqual(requests.post.call_args.kwargs['timeout'], 3.0)

    def test_local_when_endpoint_busy_with_other_model(self):
        backend, requests = _backend(running=[{'model': 'qwen', 'state': 'ready'}])
        self.assertEqual(backend.transcribe(AUDIO, 16000), 'local text')
        requests.post.assert_not_called()
        self.assertEqual(backend.last_route['served_by'], 'local')
        self.assertIn('endpoint holds qwen', backend.last_route['reason'])

    def test_local_when_unreachable(self):
        backend, requests = _backend(health_error=OSError('no route'))
        self.assertEqual(backend.transcribe(AUDIO, 16000), 'local text')
        requests.post.assert_not_called()
        self.assertEqual(backend.last_route['reason'], 'endpoint unreachable')

    def test_local_while_model_still_loading(self):
        backend, requests = _backend(running=[{'model': MODEL, 'state': 'starting'}])
        self.assertEqual(backend.transcribe(AUDIO, 16000), 'local text')
        requests.post.assert_not_called()

    def test_falls_back_when_remote_times_out(self):
        timeout = rest_api_backend.require_package('requests').exceptions.Timeout

        def post(*args, **kwargs):
            raise timeout()

        backend, _ = _backend(running=READY, post=post)
        self.assertEqual(backend.transcribe(AUDIO, 16000), 'local text')
        self.assertIn('remote failed: timed out', backend.last_route['reason'])

    def test_falls_back_on_http_error(self):
        backend, _ = _backend(
            running=READY,
            post=lambda *a, **k: mock.Mock(status_code=503, json=mock.Mock(return_value={})),
        )
        self.assertEqual(backend.transcribe(AUDIO, 16000), 'local text')

    def test_empty_remote_transcript_is_not_a_failure(self):
        backend, _ = _backend(
            running=READY,
            post=lambda *a, **k: mock.Mock(status_code=200, json=mock.Mock(return_value={'text': ''})),
        )
        self.assertEqual(backend.transcribe(AUDIO, 16000), '')
        self.assertEqual(backend.last_route['served_by'], 'remote')

    def test_without_fallback_failures_still_return_empty(self):
        backend, _ = _backend(
            settings={'rest_fallback_backend': None},
            post=lambda *a, **k: mock.Mock(status_code=500, json=mock.Mock(return_value={})),
        )
        self.assertEqual(backend.transcribe(AUDIO, 16000), '')


class RestWarmupTests(unittest.TestCase):
    def _warm(self, running, settings=None):
        backend, requests = _backend(running=running, settings=settings)
        threads = []

        class Capture:
            def __init__(self, target, **kwargs):
                threads.append(target)

            def start(self):
                pass

        with mock.patch.object(rest_api_backend.threading, 'Thread', Capture):
            backend.on_recording_started()
        for target in threads:
            target()
        return requests

    def test_warms_idle_endpoint(self):
        requests = self._warm(running=[])
        requests.post.assert_called_once()

    def test_never_evicts_another_model(self):
        requests = self._warm(running=[{'model': 'qwen', 'state': 'ready'}])
        requests.post.assert_not_called()

    def test_warms_beside_coresident_model(self):
        requests = self._warm(
            running=[{'model': 'embed', 'state': 'ready'}],
            settings={'rest_coresident_models': ['embed']},
        )
        requests.post.assert_called_once()

    def test_coresident_list_never_covers_other_models(self):
        requests = self._warm(
            running=[{'model': 'embed', 'state': 'ready'}, {'model': 'qwen', 'state': 'ready'}],
            settings={'rest_coresident_models': ['embed']},
        )
        requests.post.assert_not_called()

    def test_no_warmup_when_unreachable(self):
        backend, requests = _backend(health_error=OSError('down'))
        threads = []

        class Capture:
            def __init__(self, target, **kwargs):
                threads.append(target)

            def start(self):
                pass

        with mock.patch.object(rest_api_backend.threading, 'Thread', Capture):
            backend.on_recording_started()
        for target in threads:
            target()
        requests.post.assert_not_called()

    def test_no_warmup_when_already_loaded(self):
        requests = self._warm(running=READY)
        requests.post.assert_not_called()


if __name__ == '__main__':
    unittest.main()

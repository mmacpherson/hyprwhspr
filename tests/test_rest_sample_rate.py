import io
import sys
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends import BACKENDS  # noqa: E402


def _backend(settings):
    config = mock.Mock()
    config.get_setting.side_effect = lambda key, default=None: {
        'rest_endpoint_url': 'https://example.invalid/transcribe',
        'rest_body': {'prompt': ''},
        **settings,
    }.get(key, default)
    manager = SimpleNamespace(
        config=config, current_model='m', ready=True, temp_dir='/tmp', _last_use_time=0.0
    )
    backend = BACKENDS['rest-api'](manager)
    requests = mock.Mock()
    requests.post.return_value = mock.Mock(
        status_code=200, json=mock.Mock(return_value={'text': 'ok'})
    )
    backend._requests = requests
    return backend, requests


def _uploaded_rate(requests):
    wav_bytes = requests.post.call_args.kwargs['files']['file'][1]
    with wave.open(io.BytesIO(wav_bytes)) as wav:
        return wav.getframerate()


class RestSampleRateTests(unittest.TestCase):
    audio = np.zeros(44_100, dtype=np.float32)

    def test_downsamples_to_configured_rate(self):
        backend, requests = _backend({'rest_sample_rate': 16_000})
        self.assertEqual(backend.transcribe(self.audio, 44_100), 'ok')
        self.assertEqual(_uploaded_rate(requests), 16_000)

    def test_sends_capture_rate_by_default(self):
        backend, requests = _backend({})
        backend.transcribe(self.audio, 44_100)
        self.assertEqual(_uploaded_rate(requests), 44_100)

    def test_never_upsamples(self):
        backend, requests = _backend({'rest_sample_rate': 16_000})
        backend.transcribe(np.zeros(8_000, dtype=np.float32), 8_000)
        self.assertEqual(_uploaded_rate(requests), 8_000)


if __name__ == "__main__":
    unittest.main()

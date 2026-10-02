import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends import rest_api_backend  # noqa: E402
from transcription_summary import format_transcription_summary, short_model_name  # noqa: E402


class TranscriptionSummaryTests(unittest.TestCase):
    def test_remote_summary(self):
        route = {'served_by': 'remote', 'host': 'gpu-box', 'model': 'whisper-small-en'}
        text = format_transcription_summary(36.4, 3.21, 'one two three', route, 'laptop', 'base.en')
        self.assertEqual(text, '36s : gpu-box : small.en : 3.2s : 3w')

    def test_local_fallback_adds_reason(self):
        route = {'served_by': 'local', 'host': 'laptop', 'model': 'base.en',
                 'reason': 'endpoint holds qwen (busy)'}
        text = format_transcription_summary(5, 0.4, 'hi', route, 'laptop', 'base.en')
        self.assertEqual(text, '5s : laptop : base.en : 0.4s : 1w\nendpoint holds qwen (busy)')

    def test_plain_local_backend_uses_defaults(self):
        text = format_transcription_summary(10, 1.0, '', {}, 'laptop', 'base.en')
        self.assertEqual(text, '10s : laptop : base.en : 1.0s : 0w')

    def test_short_model_names(self):
        self.assertEqual(short_model_name('whisper-medium-en'), 'medium.en')
        self.assertEqual(short_model_name('whisper-large-v3-turbo'), 'large-v3-turbo')
        self.assertEqual(short_model_name('small.en'), 'small.en')


class HostLabelTests(unittest.TestCase):
    def test_reverse_dns_short_name(self):
        with mock.patch.object(rest_api_backend.socket, 'gethostbyaddr',
                               return_value=('gpu-box.example.ts.net', [], [])):
            self.assertEqual(rest_api_backend._host_label('http://100.64.0.7:9080/x'), 'gpu-box')

    def test_unresolvable_ip_kept_whole(self):
        with mock.patch.object(rest_api_backend.socket, 'gethostbyaddr', side_effect=OSError):
            self.assertEqual(rest_api_backend._host_label('http://10.0.0.5:9080/x'), '10.0.0.5')


if __name__ == '__main__':
    unittest.main()

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends import rest_api_backend  # noqa: E402
from transcription_summary import (  # noqa: E402
    disfluency_counts,
    format_disfluency_summary,
    format_transcription_summary,
    short_model_name,
)


class TranscriptionSummaryTests(unittest.TestCase):
    def test_remote_summary(self):
        route = {'served_by': 'remote', 'host': 'gpu-box', 'model': 'whisper-small-en'}
        text = format_transcription_summary(36.4, 3.21, 'one two three', route, 'laptop', 'base.en')
        self.assertEqual(text, '36s : gpu-box : small.en : 3.2s : 11.3x : 3w')

    def test_local_fallback_adds_reason(self):
        route = {'served_by': 'local', 'host': 'laptop', 'model': 'base.en',
                 'reason': 'endpoint holds qwen (busy)'}
        text = format_transcription_summary(5, 0.4, 'hi', route, 'laptop', 'base.en')
        self.assertEqual(text, '5s : laptop : base.en : 0.4s : 12.5x : 1w\nendpoint holds qwen (busy)')

    def test_plain_local_backend_uses_defaults(self):
        text = format_transcription_summary(10, 1.0, '', {}, 'laptop', 'base.en')
        self.assertEqual(text, '10s : laptop : base.en : 1.0s : 10.0x : 0w')

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


class DisfluencyCountTests(unittest.TestCase):
    def test_counts_filler_positions_only(self):
        counts = disfluency_counts(
            "So, uh, I was, like, thinking. I like it, you know, and do you know what? "
            "It is so good. Um, the the plan."
        )
        self.assertEqual(counts['uh'], 2)
        self.assertEqual(counts['you know'], 1)   # not "do you know what"
        self.assertEqual(counts['like'], 1)       # not "I like it"
        self.assertEqual(counts['so'], 1)         # not "so good"
        self.assertEqual(counts['rep'], 1)

    def test_summary_line(self):
        counts = {'uh': 4, 'you know': 1, 'like': 1, 'so': 0, 'rep': 0, 'words': 100}
        self.assertEqual(
            format_disfluency_summary(counts),
            'fillers 6.0/100w : uh 4 : you know 1 : like 1 : so 0 : rep 0',
        )

    def test_empty_transcript(self):
        self.assertEqual(format_disfluency_summary(disfluency_counts('')),
                         'fillers 0.0/100w : uh 0 : you know 0 : like 0 : so 0 : rep 0')


if __name__ == '__main__':
    unittest.main()

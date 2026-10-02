"""Tests for private, opt-in dictation archival."""

import json
import stat
import sys
import tempfile
import threading
import types
import unittest
import wave
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

import recording_archive

from tests.test_suspend_resume_recovery import _import_main_isolated


class FakeConfig:
    def __init__(self, values):
        self.values = values

    def get_setting(self, name, default=None):
        return self.values.get(name, default)


class RecordingArchiveTests(unittest.TestCase):
    def test_disabled_archive_does_not_create_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            with mock.patch.object(recording_archive, "DATA_DIR", data_dir):
                archived = recording_archive.archive_recording(
                    np.ones(10, dtype=np.float32),
                    16_000,
                    FakeConfig({"archive_recordings": False}),
                )

            self.assertIsNone(archived)
            self.assertFalse(data_dir.exists())

    def test_writes_private_wav_and_finalized_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_dir = Path(temporary) / "corpus"
            config = FakeConfig({
                "archive_recordings": True,
                "recording_archive_directory": str(archive_dir),
                "transcription_backend": "vulkan",
                "model": "base.en",
                "language": "en",
                "recording_mode": "auto",
            })
            archived = recording_archive.archive_recording(
                np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),
                5,
                config,
            )

            self.assertIsNotNone(archived)
            self.assertEqual(stat.S_IMODE(archive_dir.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(archived.audio_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(archived.metadata_path.stat().st_mode), 0o600)
            with wave.open(str(archived.audio_path), "rb") as input_file:
                self.assertEqual(input_file.getnchannels(), 1)
                self.assertEqual(input_file.getsampwidth(), 2)
                self.assertEqual(input_file.getframerate(), 5)
                self.assertEqual(input_file.getnframes(), 5)

            archived.finish("injected", transcript="Test dictation.")
            metadata = json.loads(archived.metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["status"], "injected")
            self.assertEqual(metadata["transcript"], "Test dictation.")
            self.assertEqual(metadata["audio"]["duration_seconds"], 1.0)
            self.assertEqual(metadata["transcription"], {
                "backend": "vulkan",
                "model": "base.en",
                "language": "en",
            })
            self.assertEqual(metadata["recording"]["mode"], "auto")
            self.assertEqual(list(archive_dir.glob("*.tmp")), [])

    def test_write_failure_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as temporary:
            blocked = Path(temporary) / "file"
            blocked.write_text("not a directory", encoding="utf-8")
            archived = recording_archive.archive_recording(
                np.ones(10, dtype=np.float32),
                16_000,
                FakeConfig({
                    "archive_recordings": True,
                    "recording_archive_directory": str(blocked / "corpus"),
                }),
            )
            self.assertIsNone(archived)


class RecordingArchiveIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_isolated()

    def test_process_audio_archives_before_transcription_and_finalizes(self):
        events = []
        archive = mock.Mock()
        app = self.main.hyprwhsprApp.__new__(self.main.hyprwhsprApp)
        app._recording_lock = threading.Lock()
        app.is_processing = False
        app._current_language_override = None
        app.config = mock.Mock()
        app.config.get_hallucination_markers.return_value = []
        app.audio_capture = types.SimpleNamespace(sample_rate=16_000)

        def transcribe(*args, **kwargs):
            events.append("transcribe")
            return "Test dictation."

        app.whisper_manager = types.SimpleNamespace(
            transcribe_audio=transcribe,
            last_transcription_route=lambda: {"served_by": "local"},
            transcribe_verbatim=lambda *args: None,
        )
        app._inject_text = mock.Mock(return_value=self.main.hyprwhsprApp._process_audio.__globals__["InjectionOutcome"].INJECTED)
        app._notify_capture = mock.Mock()
        app._clear_mic_osd_preview_text = mock.Mock()
        app._show_result_and_hide = mock.Mock()
        app._notify_user = mock.Mock()
        app._take_live_typing = mock.Mock(return_value=None)

        def capture(*args, **kwargs):
            events.append("archive")
            return archive

        # _process_audio now lives in a mixin module; patch the name it resolves.
        process_globals = self.main.hyprwhsprApp._process_audio.__globals__
        with mock.patch.dict(process_globals, {"archive_recording": capture}):
            app._process_audio(np.ones(160, dtype=np.float32))

        self.assertEqual(events, ["archive", "transcribe"])
        archive.finish.assert_called_once_with(
            "injected", transcript="Test dictation.", route={"served_by": "local"}
        )
        app._show_result_and_hide.assert_called_once_with(True)


if __name__ == "__main__":
    unittest.main()

"""Private, opt-in archival of submitted dictation audio and metadata."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from .backend_utils import normalize_backend
    from .paths import DATA_DIR
    from .processing_trace import classify_boundary_mode, classify_vad_mode
except ImportError:
    from backend_utils import normalize_backend
    from paths import DATA_DIR
    from processing_trace import classify_boundary_mode, classify_vad_mode


def _active_model(config, backend: str):
    setting = {
        "onnx-asr": "onnx_asr_model",
        "faster-whisper": "faster_whisper_model",
        "realtime-ws": "websocket_model",
        "qwen3-asr": "qwen3_asr_model",
    }.get(backend, "model")
    if backend == "rest-api":
        body = config.get_setting("rest_body", None) or {}
        return body.get("model") if isinstance(body, dict) else None
    if backend == "cohere-transcribe":
        return "CohereLabs/cohere-transcribe-03-2026"
    return config.get_setting(setting, None)


def _atomic_json_write(path: Path, payload: dict) -> None:
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise


def _atomic_wav_write(path: Path, audio_data, sample_rate: int) -> int:
    samples = np.asarray(audio_data, dtype=np.float32).reshape(-1)
    samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")

    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            with wave.open(handle, "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(sample_rate)
                output.writeframes(pcm.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temporary_path.unlink(missing_ok=True)
        raise
    return len(pcm)


@dataclass
class ArchivedRecording:
    """A captured recording whose sidecar can be finalized after ASR."""

    metadata_path: Path
    metadata: dict

    @property
    def audio_path(self) -> Path:
        return self.metadata_path.with_suffix(".wav")

    def finish(
        self,
        status: str,
        transcript: str | None = None,
        error: str | None = None,
        route: dict | None = None,
    ) -> None:
        """Finalize metadata without allowing archive I/O to break dictation."""
        try:
            self.metadata["status"] = status
            if route:
                # A routing backend may serve from a different model than the
                # one recorded at capture (e.g. a local fallback).
                self.metadata["transcription"]["route"] = route
            self.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
            if transcript is not None:
                self.metadata["transcript"] = transcript
            if error is not None:
                self.metadata["error"] = error
            _atomic_json_write(self.metadata_path, self.metadata)
        except Exception as exc:
            print(f"[ARCHIVE] Failed to update recording metadata: {exc}", flush=True)


def archive_recording(
    audio_data,
    sample_rate: int,
    config,
    language_override: str | None = None,
) -> ArchivedRecording | None:
    """Archive one submitted recording, returning a non-fatal metadata handle."""
    try:
        if not config.get_setting("archive_recordings", False):
            return None

        configured_directory = config.get_setting("recording_archive_directory", None)
        directory = (
            Path(configured_directory).expanduser()
            if configured_directory
            else DATA_DIR / "recordings"
        )
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)

        now = datetime.now(timezone.utc)
        recording_id = f"{now.strftime('%Y%m%dT%H%M%S.%fZ')}-{uuid.uuid4().hex[:8]}"
        audio_path = directory / f"{recording_id}.wav"
        metadata_path = directory / f"{recording_id}.json"
        sample_count = _atomic_wav_write(audio_path, audio_data, int(sample_rate))

        backend = normalize_backend(
            config.get_setting("transcription_backend", "pywhispercpp")
        )
        metadata = {
            "schema_version": 1,
            "id": recording_id,
            "captured_at": now.isoformat(),
            "status": "captured",
            "audio_file": audio_path.name,
            "audio": {
                "format": "pcm_s16le",
                "sample_rate_hz": int(sample_rate),
                "channels": 1,
                "samples": sample_count,
                "duration_seconds": sample_count / int(sample_rate),
            },
            "transcription": {
                "backend": backend,
                "model": _active_model(config, backend),
                "language": language_override or config.get_setting("language", None),
            },
            "recording": {
                "mode": config.get_setting("recording_mode", "toggle"),
                "boundary_mode": classify_boundary_mode(config),
                "vad_mode": classify_vad_mode(config),
            },
        }
        _atomic_json_write(metadata_path, metadata)
        print(f"[ARCHIVE] Saved recording {audio_path}", flush=True)
        return ArchivedRecording(metadata_path=metadata_path, metadata=metadata)
    except Exception as exc:
        print(f"[ARCHIVE] Failed to save recording: {exc}", flush=True)
        return None

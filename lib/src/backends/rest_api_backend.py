"""
REST API transcription backend.

Posts recorded audio as WAV to a user-configured HTTP endpoint (any
OpenAI-compatible transcription API) and returns the transcript.
Stateless: configuration is re-read on every request.
"""

import re
import socket
import threading
import time
from urllib.parse import urlparse
from types import SimpleNamespace
from typing import Optional

try:
    from ..service_log import log
except ImportError:
    from service_log import log

try:
    from ..dependencies import require_package
except ImportError:
    from dependencies import require_package

np = require_package('numpy')

try:
    from ..credential_manager import get_credential
except ImportError:
    from credential_manager import get_credential

try:
    from ..backend_utils import endpoint_key, is_valid_http_url, redact_url
except ImportError:
    from backend_utils import endpoint_key, is_valid_http_url, redact_url

from .base import TranscriptionBackend


_URL_RE = re.compile(r'[A-Za-z][A-Za-z0-9+.-]*://[^\s\'"<>()]+')
# urllib3 names the request target alone: "... with url: /path?query".
_URL_TARGET_RE = re.compile(r'(url: /[^\s?#\'"]*)[?#][^\s\'"]*')


def _safe_error(exc):
    """Exception type and message with URLs redacted for the log."""
    text = _URL_RE.sub(lambda match: redact_url(match.group(0)), str(exc))
    text = _URL_TARGET_RE.sub(r'\1', text)
    return f'{type(exc).__name__}: {text}' if text else type(exc).__name__


def _host_label(url: str) -> str:
    """Short name for an endpoint host: 'gpu-box' from an IP that reverse-resolves."""
    host = urlparse(url or '').hostname or 'remote'
    try:
        host = socket.gethostbyaddr(host)[0]
    except (OSError, UnicodeError):
        pass
    return host if host.replace('.', '').isdigit() else host.split('.')[0]


class RemoteTranscriptionError(Exception):
    """The endpoint gave no usable transcript (as opposed to an empty one)."""


class _BackendOverride:
    """Config view that names a different transcription backend."""

    def __init__(self, config, backend: str):
        self._config, self._backend = config, backend

    def get_setting(self, key, default=None):
        if key == 'transcription_backend':
            return self._backend
        return self._config.get_setting(key, default)

    def __getattr__(self, name):
        return getattr(self._config, name)


class RestApiBackend(TranscriptionBackend):
    """Remote REST endpoint backend (no local model, no model lock)."""

    name = 'rest-api'
    is_local = False

    def __init__(self, manager):
        super().__init__(manager)
        self._requests = None
        self._fallback = None
        self.last_route = None
        self._endpoint_host = None

    def _requests_client(self):
        """Load the REST-only dependency when this backend is actually used."""
        if self._requests is None:
            self._requests = require_package('requests')
        return self._requests

    def initialize(self) -> bool:
        """Configure REST API backend"""
        self._requests_client()

        # Attempt migration of API key if needed (backup in case config was loaded before migration)
        self.config.migrate_api_key_to_credential_manager()

        # Validate REST configuration
        endpoint_url = self.config.get_setting('rest_endpoint_url')

        if not endpoint_url:
            log('ERROR: REST backend selected but rest_endpoint_url not configured')
            return False

        if not is_valid_http_url(endpoint_url.strip() if isinstance(endpoint_url, str) else endpoint_url):
            log(f'WARNING: REST endpoint URL should be an http:// or https:// URL: {redact_url(endpoint_url)}')

        # Validate timeout is reasonable
        timeout = self.config.get_setting('rest_timeout', 30)
        if timeout < 1 or timeout > 300:
            log(f'WARNING: REST timeout should be between 1-300 seconds, got {timeout}')

        log(f'[BACKEND] Using REST API: {redact_url(endpoint_url)}')
        log(f'[REST] Timeout configured: {timeout}s')

        # Log user-defined config objects (sanitized)
        rest_headers = self.config.get_setting('rest_headers', {})
        rest_body = self.config.get_setting('rest_body', {})

        # Retrieve API key: prefer credential manager, fall back to config for backward compatibility
        api_key = None
        provider_id = self.config.get_setting('rest_api_provider')
        if provider_id:
            api_key = get_credential(provider_id)
            if api_key:
                log(f'[REST] API key configured (via credential manager, provider: {provider_id})')
            else:
                log(f'WARNING: [REST] Provider {provider_id} configured but API key not found in credential store')
        else:
            # Backward compatibility: check for old rest_api_key in config
            api_key = self.config.get_setting('rest_api_key')
            if api_key:
                log('[REST] API key configured (via rest_api_key - deprecated, consider migrating)')

        if rest_headers and isinstance(rest_headers, dict):
            header_count = len([k for k in rest_headers.keys() if rest_headers.get(k) is not None])
            if header_count > 0:
                log(f'[REST] Custom headers configured ({header_count} keys)')

        if rest_body and isinstance(rest_body, dict):
            body_count = len([k for k in rest_body.keys() if rest_body.get(k) is not None])
            if body_count > 0:
                log(f'[REST] Custom body fields configured ({body_count} fields)')

        fallbacks = self._resolve_endpoints(endpoint_url, warn=True)[1:]
        if fallbacks:
            log(f'[REST] Fallbacks: {", ".join(redact_url(url) for url in fallbacks)}')

        language = self.config.get_setting('language', None)
        if language:
            log(f'[REST] Language hint: {language}')

        # Explicitly set to None to avoid confusion with top-level model setting
        self.current_model = None
        self._endpoint_host = _host_label(endpoint_url)
        self._init_fallback()
        self.ready = True
        return True

    def _has_credential(self):
        """Whether requests carry a key: stored, legacy, or a custom Authorization header."""
        provider_id = self.config.get_setting('rest_api_provider')
        if provider_id:
            if get_credential(provider_id):
                return True
        elif self.config.get_setting('rest_api_key'):
            return True
        rest_headers = self.config.get_setting('rest_headers', {})
        return isinstance(rest_headers, dict) and any(
            str(key).lower() == 'authorization' and value is not None
            for key, value in rest_headers.items())

    def _resolve_endpoints(self, endpoint_url, warn=False):
        """Primary first, then usable fallbacks, in order.

        Fallbacks are mirrors that get the primary's key, headers and body.
        Entries that are not HTTP(S) URLs with a host, or repeat an earlier
        endpoint, are dropped. So is an http:// fallback behind a keyed
        https:// primary: it would send the key in cleartext. Warnings name
        the index only, since the value may embed credentials.
        """
        primary = endpoint_url.strip() if isinstance(endpoint_url, str) else endpoint_url
        endpoints = [primary]
        fallbacks = self.config.get_setting('rest_fallback_endpoint_urls', [])
        if not isinstance(fallbacks, list):
            if warn:
                log('WARNING: rest_fallback_endpoint_urls must be an array; ignoring invalid value')
            return endpoints
        if not fallbacks:
            return endpoints

        seen = {endpoint_key(primary)} if is_valid_http_url(primary) else set()
        keyed_https = (isinstance(primary, str) and primary.lower().startswith('https://')
                       and self._has_credential())
        for index, url in enumerate(fallbacks):
            url = url.strip() if isinstance(url, str) else url
            if not is_valid_http_url(url):
                reason = 'invalid'
            elif endpoint_key(url) in seen:
                reason = 'duplicate'
            elif keyed_https and url.lower().startswith('http://'):
                reason = 'plain-http (would leak the key)'
            else:
                seen.add(endpoint_key(url))
                endpoints.append(url)
                continue
            if warn:
                log(f'WARNING: [REST] Ignoring {reason} rest_fallback_endpoint_urls entry at index {index}')
        return endpoints

    def _post_transcription(self, audio_data: np.ndarray, sample_rate: int = 16000,
                            language_override: Optional[str] = None,
                            timeout: Optional[float] = None,
                            prompt: Optional[str] = None) -> str:
        """
        Transcribe audio using remote REST API endpoint

        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate of the audio data
            language_override: Optional language code to override config language

        Returns:
            Transcribed text string
        """
        requests = self._requests_client()
        try:
            # Tracks the endpoint being attempted so outer failure handlers can
            # name it safely; stays None if an error precedes the attempt loop.
            request_url = None

            # Get REST endpoint configuration
            endpoint_url = self.config.get_setting('rest_endpoint_url')
            
            # Retrieve API key: prefer credential manager, fall back to config for backward compatibility
            api_key = None
            provider_id = self.config.get_setting('rest_api_provider')
            if provider_id:
                api_key = get_credential(provider_id)
                if not api_key:
                    log(f'WARNING: [REST] Provider {provider_id} configured but API key not found in credential store')
            else:
                # Backward compatibility: check for old rest_api_key in config
                api_key = self.config.get_setting('rest_api_key')
            
            timeout = timeout or self.config.get_setting('rest_timeout', 30)
            rest_headers = self.config.get_setting('rest_headers', {})
            rest_body = self.config.get_setting('rest_body', {})

            if not isinstance(rest_headers, dict):
                log('WARNING: rest_headers must be an object/dict; ignoring invalid value')
                rest_headers = {}

            if not isinstance(rest_body, dict):
                log('WARNING: rest_body must be an object/dict; ignoring invalid value')
                rest_body = {}

            extra_headers = {}
            for key, value in rest_headers.items():
                if value is None:
                    continue
                try:
                    extra_headers[str(key)] = str(value)
                except Exception:
                    log(f'WARNING: Skipping non-serializable rest_headers entry: {key}')

            extra_body = {}
            for key, value in rest_body.items():
                if value is None:
                    continue
                try:
                    key_str = str(key)
                except Exception:
                    log(f'WARNING: Skipping rest_body entry with non-stringable key: {key}')
                    continue

                if isinstance(value, (dict, list, tuple, set)):
                    log(f'WARNING: rest_body values must be scalar (key: {key_str}); skipping entry')
                    continue

                extra_body[key_str] = value

            if not endpoint_url:
                raise ValueError('REST endpoint URL not configured')

            endpoint_urls = self._resolve_endpoints(endpoint_url)

            # Extract model information from rest_body if available (before processing)
            model_info = rest_body.get('model') if isinstance(rest_body, dict) else None

            if model_info:
                log_msg = f'[REST API] {redact_url(endpoint_url)} - model: {model_info}'
            else:
                log_msg = f'[REST API] {redact_url(endpoint_url)}'

            log(log_msg)

            # Whisper-family servers resample to 16 kHz anyway; sending the
            # capture rate (often 44.1/48 kHz) only lengthens the upload.
            target_rate = self.config.get_setting('rest_sample_rate', None)
            if target_rate and sample_rate > int(target_rate):
                audio_data = self._resample_audio(audio_data, sample_rate, int(target_rate))
                sample_rate = int(target_rate)

            # Convert audio to WAV format
            wav_bytes = self._numpy_to_wav_bytes(audio_data, sample_rate)
            audio_duration = len(audio_data) / sample_rate
            log(f'[REST] Audio: {audio_duration:.2f}s @ {sample_rate}Hz, {len(wav_bytes)} bytes')

            # Prepare the request
            files = {'file': ('audio.wav', wav_bytes, 'audio/wav')}

            headers = {'Accept': 'application/json'}
            headers.update(extra_headers)
            if api_key:
                header_names = {key.lower() for key in headers.keys()}
                if 'authorization' not in header_names:
                    headers['Authorization'] = f'Bearer {api_key}'

            # Add language parameter if configured
            data = extra_body.copy()
            # Use language_override if provided, otherwise get from config
            language = language_override if language_override is not None else self.config.get_setting('language', None)
            if language and 'language' not in data:
                data['language'] = language

            if prompt is not None:
                data['prompt'] = prompt
            # Fill prompt from config - use language-specific prompt if available
            if 'prompt' not in data:
                whisper_prompt, _ = self.resolve_whisper_prompt(language)
                if whisper_prompt:
                    data['prompt'] = whisper_prompt

            # Log request parameters for debugging
            if data:
                # Sanitize - don't log full prompt, just keys
                param_summary = ', '.join(f'{k}={v[:20] + "..." if isinstance(v, str) and len(v) > 20 else v}' for k, v in data.items())
                log(f'[REST] Request params: {param_summary}')

            # Send the request, advancing to the next endpoint only when the
            # server never got the audio (connection errors, including connect
            # timeouts) or refused it (HTTP 429/5xx). A read timeout means the
            # server may still be transcribing; retrying would bill it twice.
            for index, request_url in enumerate(endpoint_urls):
                has_fallback = index + 1 < len(endpoint_urls)
                try:
                    log(f'[REST] Sending request to {redact_url(request_url)}...')
                    start_time = time.time()
                    response = requests.post(request_url, files=files, data=data, headers=headers, timeout=timeout)
                    response_time = time.time() - start_time
                    log(f'[REST] Response received in {response_time:.2f}s (status: {response.status_code})')
                except requests.exceptions.ConnectionError as exc:
                    if not has_fallback:
                        raise
                    log(f'WARNING: [REST] Could not reach {redact_url(request_url)} '
                        f'({_safe_error(exc)}); trying fallback endpoint')
                    continue

                # Check for HTTP errors
                if response.status_code != 200:
                    retryable = response.status_code == 429 or response.status_code >= 500
                    if retryable and has_fallback:
                        log(f'WARNING: [REST] Status {response.status_code} from {redact_url(request_url)}; trying fallback endpoint')
                        # Release the discarded response body before the retry.
                        response.close()
                        continue
                    error_msg = f'REST API returned status {response.status_code}'
                    try:
                        error_detail = response.json()
                        error_msg += f': {error_detail}'
                    except Exception:
                        error_msg += f': {response.text[:200]}'
                    log(f'ERROR: {error_msg}')
                    raise RemoteTranscriptionError(error_msg)

                # Parse the response
                try:
                    result = response.json()
                except Exception as json_err:
                    # Show raw response for debugging
                    raw_body = response.text[:500] if response.text else '(empty)'
                    log(f'ERROR: Failed to parse JSON response: {json_err}')
                    log(f'[REST] Raw response body: {raw_body}')
                    log(f'[REST] Content-Type: {response.headers.get("Content-Type", "not set")}')
                    raise RemoteTranscriptionError('unparseable response')

                # Try common response formats
                transcription = ''
                if 'text' in result:
                    transcription = result['text']
                elif 'transcription' in result:
                    transcription = result['transcription']
                elif 'result' in result:
                    transcription = result['result']
                else:
                    log(f'ERROR: Unexpected response format: {result}')
                    raise RemoteTranscriptionError('unexpected response format')

                log(f'[REST] Transcription received ({len(transcription)} chars)')
                # Servers such as whisper-server break lines between segments; a
                # pasted newline can act as Enter, so join like the local backends.
                return ' '.join(transcription.split())

        except RemoteTranscriptionError:
            raise
        except requests.exceptions.Timeout:
            log(f'ERROR: REST API request timed out after {timeout}s at {redact_url(request_url) or "<unknown>"}')
            raise RemoteTranscriptionError(f'timed out after {timeout:.1f}s')
        except requests.exceptions.ConnectionError as exc:
            log(f'ERROR: Failed to connect to REST API at {redact_url(request_url) or "<unknown>"}: '
                f'{_safe_error(exc)}')
            raise RemoteTranscriptionError('connection failed')
        except requests.exceptions.RequestException as exc:
            log(f'ERROR: REST API request failed at {redact_url(request_url) or "<unknown>"}: '
                f'{_safe_error(exc)}')
            raise RemoteTranscriptionError('request failed')
        except Exception as exc:
            log(f'ERROR: REST transcription failed: {_safe_error(exc)}')
            raise RemoteTranscriptionError(type(exc).__name__)

    # ------------------------------------------------------------------
    # Local fallback
    #
    # With rest_fallback_backend set, a local backend stays loaded beside the
    # endpoint. A dictation goes remote only when the endpoint can take it now;
    # otherwise, or when the remote call fails or misses its deadline, the same
    # audio is transcribed locally. rest_health_url (llama-swap's /running)
    # answers "can it take it now" without waiting behind another model.
    # ------------------------------------------------------------------

    def _init_fallback(self) -> None:
        name = self.config.get_setting('rest_fallback_backend', None)
        if not name:
            return
        from . import BACKENDS
        from .pywhispercpp_backend import PywhispercppBackend
        try:
            from ..backend_utils import normalize_backend
        except ImportError:
            from backend_utils import normalize_backend

        name = normalize_backend(name)
        manager = SimpleNamespace(
            config=_BackendOverride(self.config, name),
            temp_dir=self.temp_dir,
            ready=False,
            current_model=None,
            _last_use_time=0.0,
        )
        fallback = BACKENDS.get(name, PywhispercppBackend)(manager)
        if fallback.initialize():
            self._fallback = fallback
            # Resume recovery must refresh the local model's GPU context.
            self.reinit_on_resume = fallback.reinit_on_resume
            log(f'[REST] Local fallback ready: {name} ({manager.current_model})')
        else:
            log(f'WARNING: [REST] Local fallback {name} failed to load; no fallback')

    def _remote_model(self) -> Optional[str]:
        body = self.config.get_setting('rest_body', {}) or {}
        return body.get('model') if isinstance(body, dict) else None

    def _running_models(self) -> Optional[list]:
        """llama-swap's loaded models, or None when the health check fails."""
        url = self.config.get_setting('rest_health_url', None)
        try:
            response = self._requests_client().get(url, timeout=0.5)
            response.raise_for_status()
            return response.json().get('running', [])
        except Exception:
            return None

    def _choose_route(self) -> tuple:
        """('remote' | 'local', reason)."""
        if not self.config.get_setting('rest_health_url', None):
            return 'remote', 'no health check'
        running = self._running_models()
        if running is None:
            return 'local', 'endpoint unreachable'
        model = self._remote_model()
        for entry in running:
            if entry.get('model') == model:
                if entry.get('state') == 'ready':
                    return 'remote', 'model loaded'
                return 'local', f"{model} {entry.get('state', 'not ready')}"
        if running:
            others = sorted(str(entry.get('model')) for entry in running)
            states = ', '.join(f'{name} ({self._activity(name)})' for name in others)
            return 'local', f'endpoint holds {states}'
        return 'local', f'{model} not loaded'

    def _activity(self, model: str) -> str:
        """'busy' or 'idle' from a loaded llama-server's metrics, else 'unknown'.

        Recorded only: one sample can't tell an idle model from a batch job
        between requests, so it never justifies evicting that model.
        """
        url = self.config.get_setting('rest_health_url', '').rsplit('/running', 1)[0]
        try:
            response = self._requests_client().get(f'{url}/upstream/{model}/metrics', timeout=0.3)
            for line in response.text.splitlines():
                if line.startswith('llamacpp:requests_processing'):
                    return 'busy' if float(line.split()[-1]) > 0 else 'idle'
        except Exception:
            pass
        return 'unknown'

    def on_recording_started(self) -> None:
        """Warm the remote model while the user talks, if that evicts nothing.

        Warms only when the endpoint is empty or holds only models listed in
        rest_coresident_models (ones the server runs beside it). Never warms
        over anything else: that would evict whatever is using the GPU, and
        the fallback covers this dictation anyway.
        """
        if self._fallback is None or not self.config.get_setting('rest_health_url', None):
            return

        model = self._remote_model()
        coresident = set(self.config.get_setting('rest_coresident_models', None) or [])

        def warm():
            running = self._running_models()
            if running is None:
                return
            loaded = {entry.get('model') for entry in running}
            # Loading our model must evict nothing: warm only over models the
            # server keeps resident beside it (llama-swap's matrix sets).
            if model in loaded or not loaded <= coresident:
                return
            silence = np.zeros(16000, dtype=np.float32)
            try:
                self._post_transcription(silence, 16000, timeout=60)
                log('[REST] Remote model warmed')
            except RemoteTranscriptionError as e:
                log(f'[REST] Warm-up failed: {e}')

        threading.Thread(target=warm, name='rest-warmup', daemon=True).start()

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000,
                   language_override: Optional[str] = None) -> str:
        """Transcribe remotely, or locally when the endpoint can't take it now."""
        remote_model = self._remote_model()
        if self._fallback is None:
            self.last_route = {'served_by': 'remote', 'host': self._endpoint_host, 'model': remote_model}
            try:
                return self._post_transcription(audio_data, sample_rate, language_override)
            except RemoteTranscriptionError:
                return ''

        route, reason = self._choose_route()
        if route == 'remote':
            duration = len(audio_data) / sample_rate
            deadline = (
                float(self.config.get_setting('rest_fallback_timeout_base', 2.0))
                + float(self.config.get_setting('rest_fallback_timeout_per_second', 0.1)) * duration
            )
            try:
                text = self._post_transcription(
                    audio_data, sample_rate, language_override, timeout=deadline
                )
                self.last_route = {'served_by': 'remote', 'host': self._endpoint_host, 'model': remote_model, 'reason': reason}
                return text
            except RemoteTranscriptionError as e:
                reason = f'remote failed: {e}'

        log(f'[REST] Transcribing locally ({reason})')
        fallback = self._fallback
        idle = time.monotonic() - fallback._last_use_time
        if fallback.reinit_on_idle and fallback._last_use_time > 0 and idle > 1800:
            fallback.reinitialize()
        self.last_route = {'served_by': 'local', 'host': socket.gethostname(), 'model': fallback.current_model, 'reason': reason}
        return fallback.transcribe(audio_data, sample_rate, language_override=language_override)

    def reinitialize(self) -> bool:
        return self._fallback.reinitialize() if self._fallback is not None else True

    def cleanup(self) -> None:
        if self._fallback is not None:
            self._fallback.cleanup()

    def transcribe_verbatim(self, audio_data: np.ndarray, sample_rate: int, prompt: str) -> Optional[str]:
        """A second, prompted pass that keeps fillers; None if the endpoint can't serve.

        Only when the endpoint is ready now: this pass is for statistics, so it
        never falls back locally or waits behind another model.
        """
        if self._choose_route()[0] != 'remote':
            return None
        duration = len(audio_data) / sample_rate
        deadline = (
            float(self.config.get_setting('rest_fallback_timeout_base', 2.0))
            + float(self.config.get_setting('rest_fallback_timeout_per_second', 0.1)) * duration
        )
        try:
            return self._post_transcription(audio_data, sample_rate, timeout=deadline, prompt=prompt)
        except RemoteTranscriptionError:
            return None

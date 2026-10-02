"""One-line summary of a transcription for the optional notification."""

import re


def short_model_name(model) -> str:
    """'whisper-small-en' and 'small.en' both read as 'small.en'."""
    name = re.sub(r'^whisper-', '', str(model or '?'))
    return re.sub(r'-en$', '.en', name)


def format_transcription_summary(
    audio_seconds: float,
    elapsed_seconds: float,
    transcription,
    route: dict,
    default_host: str,
    default_model,
) -> str:
    """'36s : gpu-box : small.en : 3.2s : 417w', plus why when it fell back."""
    words = len((transcription or '').split())
    summary = ' : '.join([
        f'{audio_seconds:.0f}s',
        route.get('host') or default_host,
        short_model_name(route.get('model') or default_model),
        f'{elapsed_seconds:.1f}s',
        f'{words}w',
    ])
    if route.get('served_by') == 'local' and route.get('reason'):
        summary += f"\n{route['reason']}"
    return summary

"""One-line summaries of a transcription for the optional notifications."""

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
    """'36s : gpu-box : small.en : 3.2s : 11.2x : 417w' (x: speed over real time), plus why when it fell back."""
    words = len((transcription or '').split())
    summary = ' : '.join([
        f'{audio_seconds:.0f}s',
        route.get('host') or default_host,
        short_model_name(route.get('model') or default_model),
        f'{elapsed_seconds:.1f}s',
        f'{audio_seconds / max(elapsed_seconds, 1e-3):.1f}x',
        f'{words}w',
    ])
    if route.get('served_by') == 'local' and route.get('reason'):
        summary += f"\n{route['reason']}"
    return summary


# Patterns over a verbatim (filler-keeping) transcript. "like" and "so" are
# mostly ordinary words, so only their filler positions count: "like" next to
# a comma, "so" opening a sentence.
_DISFLUENCIES = {
    'uh': re.compile(r'\b(?:uh+|um+|uhm|er+m?|ah+|hmm+)\b', re.IGNORECASE),
    'you know': re.compile(
        r"(?<!\bdo )(?<!\bdid )(?<!\bdon't )(?<!\bif )(?<!\bas )(?<!\bwhat )(?<!\bhow )"
        r"\byou know\b(?! (?:what|how|that|the|about|if|why|where|who)\b)",
        re.IGNORECASE,
    ),
    'like': re.compile(r'(?:^|[,.!?])\s*like\b|\blike(?=,)', re.IGNORECASE),
    'so': re.compile(r'(?:^|[.!?]\s+)so\b', re.IGNORECASE),
    'rep': re.compile(r'\b(?!(?:that|had)\b)(\w+)[,.]? \1\b', re.IGNORECASE),
}


def disfluency_counts(verbatim: str) -> dict:
    """Counts per disfluency in a filler-keeping transcript, plus total words."""
    counts = {name: len(pattern.findall(verbatim or '')) for name, pattern in _DISFLUENCIES.items()}
    counts['words'] = len((verbatim or '').split())
    return counts


def format_disfluency_summary(counts: dict) -> str:
    """'fillers 6.1/100w : uh 4 : you know 1 : like 1 : so 0 : rep 0'."""
    fillers = sum(v for k, v in counts.items() if k != 'words')
    rate = 100 * fillers / max(counts['words'], 1)
    parts = [f'fillers {rate:.1f}/100w'] + [
        f'{name} {counts[name]}' for name in _DISFLUENCIES
    ]
    return ' : '.join(parts)

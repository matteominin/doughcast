"""Local Italian transcription with faster-whisper."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dough.config import WHISPER_MODEL

ITALIAN_PIZZERIA_PROMPT = (
    "Trascrizione di una nota di una pizzeria italiana: impasto, panetti, "
    "pizze, coperti, finito, asporto, delivery, margherita, vendute, "
    "rimaste, partita, pioggia."
)


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start: float | None = None
    end: float | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class Transcription:
    text: str
    segments: list[TranscriptSegment]


def _load_model(model_size: str) -> Any:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - optional runtime
        raise RuntimeError("faster-whisper non è installato") from exc
    return WhisperModel(model_size)


def transcribe_audio(
    audio_path: str | Path,
    *,
    model_size: str | None = None,
    model: Any | None = None,
    initial_prompt: str = ITALIAN_PIZZERIA_PROMPT,
) -> Transcription:
    """Transcribe an audio path locally in Italian.

    faster-whisper delegates webm/ogg/wav/m4a decoding to ffmpeg. The path is
    read for this call only; audio is not copied to an application directory.
    """
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    whisper = model or _load_model(model_size or WHISPER_MODEL)
    segments, _info = whisper.transcribe(
        str(path), language="it", initial_prompt=initial_prompt
    )
    collected: list[TranscriptSegment] = []
    for segment in segments:
        avg_logprob = getattr(segment, "avg_logprob", None)
        confidence = None if avg_logprob is None else min(1.0, max(0.0, 1.0 + avg_logprob / 5.0))
        collected.append(
            TranscriptSegment(
                text=str(getattr(segment, "text", "")).strip(),
                start=getattr(segment, "start", None),
                end=getattr(segment, "end", None),
                confidence=confidence,
            )
        )
    return Transcription(" ".join(part.text for part in collected).strip(), collected)


def transcribe(*args: Any, **kwargs: Any) -> Transcription:
    """Backward-compatible alias for :func:`transcribe_audio`."""
    return transcribe_audio(*args, **kwargs)
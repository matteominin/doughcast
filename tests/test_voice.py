from datetime import date
from types import SimpleNamespace

import pytest

from dough.voice.extract import extract_transcript, parse_transcript_for_eval
from dough.voice.transcribe import transcribe_audio


class FakeOllama:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {"message": {"content": self.content}}


def test_extraction_uses_structured_format_and_validates_warnings():
    client = FakeOllama(
        '{"date":"2026-10-02","pizzas_sold":74,"dough_balls_prepared":90,'
        '"dough_balls_left":2,"sold_out":true,"sold_out_time":null,"closed":null,'
        '"weather":"rain","events":["partita"],"notes":null}'
    )
    result = extract_transcript("nota", reference_date=date(2026, 10, 2), client=client)

    assert result.extracted.pizzas_sold == 74
    assert any("non tornano" in warning for warning in result.warnings)
    assert client.calls[0]["format"]["type"] == "object"
    assert client.calls[0]["messages"][1]["content"].startswith("Data di riferimento:")


def test_invalid_ollama_json_retries_once_then_returns_empty_record():
    client = FakeOllama("not-json")
    result = extract_transcript("nota", reference_date=date(2026, 10, 2), client=client)

    assert len(client.calls) == 2
    assert result.extracted.pizzas_sold is None
    assert "due tentativi" in result.warnings[0]


def test_recalled_mode_returns_list_and_marks_source():
    client = FakeOllama(
        '[{"date":"2026-09-28","pizzas_sold":45,"dough_balls_prepared":null,'
        '"dough_balls_left":null,"sold_out":null,"sold_out_time":null,"closed":null,'
        '"weather":null,"events":null,"notes":null}]'
    )
    result = extract_transcript("domenica scorsa 45", reference_date=date(2026, 10, 2), mode="recalled", client=client)

    assert isinstance(result.extracted, list)
    assert result.extracted[0].source == "recalled"
    assert client.calls[0]["format"]["type"] == "array"


def test_transcriber_is_injectable_and_preserves_segment_confidence(tmp_path):
    audio = tmp_path / "note.wav"
    audio.write_bytes(b"fake")

    class FakeModel:
        def transcribe(self, path, **kwargs):
            assert path == str(audio)
            assert kwargs["language"] == "it"
            assert "impasto" in kwargs["initial_prompt"]
            return [SimpleNamespace(text=" vendute 74 pizze ", start=0.0, end=1.2, avg_logprob=-0.5)], None

    result = transcribe_audio(audio, model=FakeModel())
    assert result.text == "vendute 74 pizze"
    assert result.segments[0].confidence == pytest.approx(0.9)


def test_deterministic_parser_keeps_unsaid_fields_null():
    result = parse_transcript_for_eval("Che stanchezza oggi, domani vediamo.", date(2026, 10, 2))
    assert result.pizzas_sold is None
    assert result.events is None
    assert result.weather is None


@pytest.mark.slow
@pytest.mark.skip(reason="Requires downloaded faster-whisper weights and ffmpeg")
def test_real_whisper_integration():
    pass

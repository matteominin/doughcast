"""Structured, local extraction of Italian pizzeria notes through Ollama."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from dough.config import OLLAMA_MODEL
from dough.schemas import DailyRecord

PROMPT_PATH = Path(__file__).resolve().parents[3] / "prompts" / "extract_it.txt"
MAX_PIZZAS = int(os.getenv("EXTRACTION_MAX_PIZZAS", "1000"))
ARITHMETIC_TOLERANCE = int(os.getenv("EXTRACTION_ARITHMETIC_TOLERANCE", "2"))

DAILY_RECORD_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "date": {"type": ["string", "null"]}, "pizzas_sold": {"type": ["integer", "null"]},
        "dough_balls_prepared": {"type": ["integer", "null"]}, "dough_balls_left": {"type": ["integer", "null"]},
        "sold_out": {"type": ["boolean", "null"]}, "sold_out_time": {"type": ["string", "null"]},
        "closed": {"type": ["boolean", "null"]},
        "weather": {"type": ["string", "null"], "enum": ["clear", "cloudy", "rain", "snow", None]},
        "events": {"type": ["array", "null"], "items": {"type": "string"}},
        "notes": {"type": ["string", "null"]},
    },
    "required": ["date", "pizzas_sold", "dough_balls_prepared", "dough_balls_left", "sold_out",
                 "sold_out_time", "closed", "weather", "events", "notes"],
}


@dataclass(frozen=True)
class ExtractionResult:
    extracted: DailyRecord | list[DailyRecord]
    warnings: list[str]


def _system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:  # pragma: no cover
        return "Rispondi solo con JSON conforme allo schema DailyRecord."


def _empty_record(reference_date: date, source: str) -> DailyRecord:
    return DailyRecord(date=reference_date, source=source, confirmed=False)


def _response_content(response: Any) -> str:
    message = response.get("message", {}) if isinstance(response, dict) else getattr(response, "message", {})
    return str(message.get("content", "") if isinstance(message, dict) else getattr(message, "content", ""))


def _validate(payload: dict[str, Any], reference_date: date, source: str) -> tuple[DailyRecord, list[str]]:
    warnings: list[str] = []
    payload = dict(payload)
    if payload.get("date") is None:
        warnings.append("La data non è stata detta chiaramente; uso la data di riferimento.")
    payload.setdefault("date", reference_date.isoformat())
    payload["source"], payload["confirmed"] = source, False
    try:
        record = DailyRecord.model_validate(payload)
    except ValidationError as exc:
        return _empty_record(reference_date, source), ["La risposta non contiene dati validi: " + str(exc)]
    if record.pizzas_sold is not None and record.pizzas_sold > MAX_PIZZAS:
        warnings.append(f"Il numero di pizze ({record.pizzas_sold}) è oltre il limite plausibile configurato.")
    if record.dough_balls_left is not None and record.dough_balls_prepared is not None and record.dough_balls_left > record.dough_balls_prepared:
        warnings.append("Le palline rimaste sono più delle palline preparate.")
    if all(value is not None for value in (record.pizzas_sold, record.dough_balls_prepared, record.dough_balls_left)):
        difference = record.dough_balls_prepared - record.dough_balls_left
        if abs(difference - record.pizzas_sold) > ARITHMETIC_TOLERANCE:
            warnings.append("Le pizze vendute non tornano con palline preparate e rimaste.")
    if record.sold_out and record.sold_out_time is None:
        warnings.append("È indicato che l'impasto è finito, ma non è indicata l'ora.")
    return record, warnings


def _make_client(client: Any | None) -> Any:
    if client is not None:
        return client
    try:
        import ollama
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Il client Ollama non è installato") from exc
    return ollama.Client()


def extract_transcript(transcript: str, *, reference_date: date, mode: str = "daily",
                       client: Any | None = None, model_name: str = OLLAMA_MODEL) -> ExtractionResult:
    """Extract one record or recalled records; failures become UI warnings."""
    if mode not in {"daily", "recalled"}:
        raise ValueError("mode deve essere daily oppure recalled")
    source = "recalled" if mode == "recalled" else "voice"
    schema = DAILY_RECORD_SCHEMA if mode == "daily" else {"type": "array", "items": DAILY_RECORD_SCHEMA}
    user_prompt = f"Data di riferimento: {reference_date.isoformat()}\nModalità: {mode}\nNota:\n{transcript}"
    try:
        ollama_client = _make_client(client)
    except RuntimeError as exc:
        return ExtractionResult(_empty_record(reference_date, source), [str(exc)])
    messages = [{"role": "system", "content": _system_prompt()}, {"role": "user", "content": user_prompt}]
    last_error = ""
    for _attempt in range(2):
        try:
            response = ollama_client.chat(model=model_name, messages=messages, format=schema)
            parsed = json.loads(_response_content(response))
            if mode == "daily":
                record, warnings = _validate(parsed, reference_date, source)
                return ExtractionResult(record, warnings)
            if not isinstance(parsed, list):
                raise ValueError("la risposta richiamata non è una lista")
            records, warnings = [], []
            for item in parsed:
                record, item_warnings = _validate(item, reference_date, source)
                records.append(record)
                warnings.extend(item_warnings)
            return ExtractionResult(records, warnings)
        except Exception as exc:
            last_error = str(exc)
    return ExtractionResult(_empty_record(reference_date, source), [f"Estrazione non disponibile dopo due tentativi: {last_error}"])


_NUMBER_WORDS = {"zero": 0, "uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5,
                 "sei": 6, "sette": 7, "otto": 8, "nove": 9, "dieci": 10, "venti": 20,
                 "trenta": 30, "quaranta": 40, "cinquanta": 50, "sessanta": 60, "settanta": 70,
                 "ottanta": 80, "novanta": 90, "cento": 100, "quarantacinque": 45,
                 "cinquantadue": 52, "sessantotto": 68, "settantaquattro": 74,
                 "settantasei": 76, "ottantadue": 82, "novantacinque": 95,
                 "dodici": 12}


def _number_match(text: str) -> int | None:
    match = re.search(r"\b(\d{1,4})\b", text)
    if match:
        return int(match.group(1))
    for word, number in sorted(_NUMBER_WORDS.items(), key=lambda item: -len(item[0])):
        if re.search(rf"\b{word}\b", text.lower()):
            return number
    return None


def parse_transcript_for_eval(text: str, reference_date: date) -> DailyRecord:
    """Deterministic parser for offline evaluation; never used in production."""
    lowered = text.lower()
    values: dict[str, Any] = {"date": reference_date, "source": "voice", "confirmed": False}
    if any(token in lowered for token in ("chiuso", "chiusa", "chiusi", "non abbiamo aperto")):
        values["closed"] = True
    if "finito l'impasto" in lowered or "finite le palline" in lowered or "finito le palline" in lowered:
        values["sold_out"] = True
    factual_weather = "parlare di" not in lowered
    if "piov" in lowered and factual_weather:
        values["weather"] = "rain"
    elif "nev" in lowered and factual_weather:
        values["weather"] = "snow"
    elif "nuvol" in lowered and factual_weather:
        values["weather"] = "cloudy"
    elif ("sereno" in lowered or "serena" in lowered or "sole" in lowered) and factual_weather:
        values["weather"] = "clear"
    time_match = re.search(r"(?:alle|ore)\s+(\d{1,2})(?::| e )?(\d{2})?", lowered)
    if time_match and values.get("sold_out"):
        values["sold_out_time"] = f"{int(time_match.group(1)):02d}:{int(time_match.group(2) or 0):02d}"
    sold_match = re.search(r"(?:vendut[oeai]|fatte|sfornat[ei]|pizze)\D{0,20}(\d{1,4}|[a-zà]+)", lowered)
    sold_number = _number_match(sold_match.group(0)) if sold_match else None
    if sold_number is None:
        sold_match = re.search(r"(\d{1,4}|[a-zà]+)\s+pizze", lowered)
        sold_number = _number_match(sold_match.group(0)) if sold_match else None
    if sold_number is not None:
        values["pizzas_sold"] = sold_number
    prepared_match = re.search(r"(?:preparat[oei]|impastat[oei]|palline)\D{0,20}(\d{1,4}|[a-zà]+)", lowered)
    prepared_number = _number_match(prepared_match.group(0)) if prepared_match else None
    if prepared_number is None:
        prepared_match = re.search(r"(\d{1,4}|[a-zà]+)\s+(?:palline|panetti)", lowered)
        prepared_number = _number_match(prepared_match.group(0)) if prepared_match else None
    if prepared_number is not None:
        values["dough_balls_prepared"] = prepared_number
    left_match = re.search(r"(?:rimast[ei]|avanzat[ei])\D{0,20}(\d{1,4}|[a-zà]+)", lowered)
    left_number = _number_match(left_match.group(0)) if left_match else None
    if left_number is None:
        left_match = re.search(r"(\d{1,4}|[a-zà]+)\s+(?:rimast[ei]|avanzat[ei])", lowered)
        left_number = _number_match(left_match.group(0)) if left_match else None
    if left_number is not None:
        values["dough_balls_left"] = left_number
    if "partita" in lowered and factual_weather:
        values["events"] = ["partita"]
    if "asporto" in lowered or "delivery" in lowered:
        values["events"] = (values.get("events") or []) + ["asporto/delivery"]
    return DailyRecord.model_validate(values)


def extract(*args: Any, **kwargs: Any) -> ExtractionResult:
    return extract_transcript(*args, **kwargs)
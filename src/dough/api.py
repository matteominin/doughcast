"""Local FastAPI application for records, voice notes, and forecasts."""

from __future__ import annotations

import inspect
import os
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from dough.config import (
    HORIZON_DAYS,
    MIN_TRAIN_ROWS,
    OLLAMA_MODEL,
    TABPFN_VERSION,
    USE_RECALLED_IN_FIT,
    WHISPER_MODEL,
)
from dough.data.features import effective_rows, build_features
from dough.forecast.dough import recommendation_from_forecast
from dough.forecast.tabpfn_model import TabPFNModel
from dough.schemas import (
    DailyRecord,
    Forecast,
    HealthResponse,
    RecordsResponse,
    StatusResponse,
    VoiceResponse,
)
from dough.store import RecordStore


@dataclass
class _AppState:
    store: RecordStore
    transcriber: Any | None = None
    extractor: Any | None = None
    forecaster: Any | None = None
    voice_loading: bool = False
    forecast_loading: bool = False


def _call_with_supported_kwargs(function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call simple test fakes as well as the production adapters."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    return function(*args, **accepted)


def _records_for_fit(state: _AppState) -> list[DailyRecord]:
    records = state.store.list_records()
    return [
        record
        for record in records
        if record.confirmed is True
        and record.closed is not True
        and record.pizzas_sold is not None
        and (USE_RECALLED_IN_FIT or record.source != "recalled")
    ]


def _effective_count(records: list[DailyRecord]) -> int:
    if not records:
        return 0
    import pandas as pd

    frame = pd.DataFrame(
        [{"date": record.date, "pizzas_sold": record.pizzas_sold} for record in records]
    )
    return len(effective_rows(build_features(frame, horizon_days=HORIZON_DAYS), "pizzas_sold"))


def _get_transcriber(state: _AppState) -> Any:
    if state.transcriber is not None:
        return state.transcriber
    state.voice_loading = True
    try:
        from dough.voice.transcribe import transcribe_audio

        state.transcriber = transcribe_audio
        return state.transcriber
    finally:
        state.voice_loading = False


def _get_extractor(state: _AppState) -> Any:
    if state.extractor is not None:
        return state.extractor
    state.voice_loading = True
    try:
        from dough.voice.extract import extract_transcript

        state.extractor = extract_transcript
        return state.extractor
    finally:
        state.voice_loading = False


def _get_forecaster(state: _AppState) -> Any:
    if state.forecaster is not None:
        return state.forecaster
    state.forecast_loading = True
    try:
        state.forecaster = TabPFNModel()
        return state.forecaster
    finally:
        state.forecast_loading = False


def _transcript_text(value: Any) -> str:
    return value.text if hasattr(value, "text") else str(value)


def _voice_result(value: Any, reference_date: date, mode: str) -> tuple[Any, list[str]]:
    if hasattr(value, "extracted"):
        return value.extracted, list(value.warnings)
    if isinstance(value, tuple) and len(value) == 2:
        return value[0], list(value[1])
    return value, []


def _prediction_to_forecast(prediction: Any, target_date: date, history_size: int) -> Forecast:
    if isinstance(prediction, Forecast):
        return prediction
    return Forecast(
        target_date=target_date,
        prepare_on=target_date - timedelta(days=HORIZON_DAYS),
        horizon_days=HORIZON_DAYS,
        p10=float(prediction.p10),
        p50=float(prediction.p50),
        p90=float(prediction.p90),
        model="tabpfn",
        n_train_days=history_size,
        baseline_p50=None,
        recommendation={
            "service_level_quantile": 0.8,
            "dough_balls": 0,
            "dough_grams": 0,
            "reasoning": "",
        },
    )


def create_app(
    *,
    store: RecordStore | None = None,
    transcriber: Any | None = None,
    extractor: Any | None = None,
    forecaster: Any | None = None,
) -> FastAPI:
    state = _AppState(
        store=store or RecordStore(os.getenv("DATABASE_PATH", "data/private/pizzeria.db")),
        transcriber=transcriber,
        extractor=extractor,
        forecaster=forecaster,
    )
    application = FastAPI(title="Doughcast")

    if os.getenv("DOUGHCAST_ALLOW_LAN", "0").lower() in {"1", "true", "yes", "on"}:
        origins = [origin.strip() for origin in os.getenv("DOUGHCAST_CORS_ORIGINS", "*").split(",")]
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    static_dir = Path(__file__).resolve().parents[2] / "static"
    if static_dir.is_dir():
        application.mount("/static", StaticFiles(directory=static_dir), name="static")

    @application.get("/", include_in_schema=False, response_model=None)
    def index() -> FileResponse | RedirectResponse:
        index_path = static_dir / "index.html"
        if index_path.is_file():
            return FileResponse(index_path)
        return RedirectResponse("/static/")

    @application.post("/api/voice", response_model=VoiceResponse)
    async def voice(
        audio: UploadFile = File(...),
        mode: Literal["daily", "recalled"] = Query("daily"),
    ) -> VoiceResponse:
        suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
        temporary_path: str | None = None
        try:
            contents = await audio.read()
            if not contents:
                raise HTTPException(status_code=400, detail="Il file audio è vuoto")
            with tempfile.NamedTemporaryFile(prefix="doughcast-", suffix=suffix, delete=False) as handle:
                handle.write(contents)
                temporary_path = handle.name
            transcriber_function = _get_transcriber(state)
            transcription = _call_with_supported_kwargs(transcriber_function, temporary_path)
            transcript = _transcript_text(transcription)
            extraction_function = _get_extractor(state)
            extracted_value = _call_with_supported_kwargs(
                extraction_function,
                transcript,
                reference_date=date.today(),
                mode=mode,
            )
            extracted, warnings = _voice_result(extracted_value, date.today(), mode)
            if mode == "recalled":
                if not isinstance(extracted, list):
                    raise HTTPException(status_code=502, detail="L'estrazione richiamata non ha restituito una lista")
                extracted = [
                    DailyRecord.model_validate(item).model_copy(update={"source": "recalled", "confirmed": False})
                    for item in extracted
                ]
            else:
                extracted = DailyRecord.model_validate(extracted).model_copy(update={"confirmed": False})
            return VoiceResponse(transcript=transcript, extracted=extracted, warnings=warnings)
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Elaborazione audio non disponibile: {error}") from error
        finally:
            if temporary_path:
                try:
                    Path(temporary_path).unlink(missing_ok=True)
                except OSError:
                    pass

    @application.post("/api/records", response_model=DailyRecord)
    def save_record(record: DailyRecord) -> DailyRecord:
        confirmed = record.model_copy(update={"confirmed": True, "source": record.source or "manual"})
        return state.store.upsert(confirmed)

    @application.get("/api/records", response_model=RecordsResponse)
    def records(from_date: date | None = Query(None, alias="from"), to_date: date | None = Query(None, alias="to")) -> RecordsResponse:
        if from_date and to_date and from_date > to_date:
            raise HTTPException(status_code=422, detail="from deve essere precedente o uguale a to")
        return RecordsResponse(records=state.store.list_records(from_date, to_date))

    @application.get("/api/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        available = _effective_count(_records_for_fit(state))
        return StatusResponse(rows_available=available, rows_required=MIN_TRAIN_ROWS, ready=available >= MIN_TRAIN_ROWS)

    @application.get("/api/forecast")
    def forecast(target: date = Query(..., alias="date")) -> dict[str, Any]:
        records_for_fit = _records_for_fit(state)
        available = _effective_count(records_for_fit)
        if available < MIN_TRAIN_ROWS:
            return {"status": "collecting_data", "rows_available": available, "rows_required": MIN_TRAIN_ROWS}
        import pandas as pd

        frame = pd.DataFrame([{"date": item.date, "pizzas_sold": item.pizzas_sold} for item in records_for_fit])
        try:
            model = _get_forecaster(state)
            fitted = model.fit(frame)
            prediction = fitted.predict(target)
            result = _prediction_to_forecast(prediction, target, available)
            return recommendation_from_forecast(result).model_dump(mode="json")
        except Exception as error:
            raise HTTPException(status_code=503, detail=f"Forecast non disponibile: {error}") from error

    @application.get("/api/history")
    def history() -> dict[str, list[dict[str, Any]]]:
        real = [record.model_dump(mode="json") for record in state.store.list_records()]
        return {"real": real, "predicted": []}

    @application.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        warming = state.voice_loading or state.forecast_loading
        return HealthResponse(
            status="warming_up" if warming else "ok",
            models_loaded=state.transcriber is not None and state.extractor is not None and state.forecaster is not None,
            versions={"ollama_model": OLLAMA_MODEL, "whisper": WHISPER_MODEL, "tabpfn": TABPFN_VERSION},
            offline=True,
        )

    return application


app = create_app()
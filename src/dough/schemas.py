"""Pydantic contracts shared by storage, voice, forecasting, and the API."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Weather = Literal["clear", "cloudy", "rain", "snow"]
RecordSource = Literal["voice", "manual", "recalled", "public_dataset"]


class DailyRecord(BaseModel):
    """One confirmed or proposed pizzeria day."""

    model_config = ConfigDict(extra="forbid")

    date: date
    pizzas_sold: int | None = Field(default=None, ge=0)
    dough_balls_prepared: int | None = Field(default=None, ge=0)
    dough_balls_left: int | None = Field(default=None, ge=0)
    sold_out: bool | None = None
    sold_out_time: str | None = None
    closed: bool | None = None
    weather: Weather | None = None
    events: list[str] | None = None
    notes: str | None = None
    source: RecordSource | None = None
    confirmed: bool | None = None

    @field_validator("sold_out_time")
    @classmethod
    def validate_time(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) != 5 or value[2] != ":":
            raise ValueError("sold_out_time must use HH:MM format")
        hour, minute = value[:2], value[3:]
        if not (hour.isdigit() and minute.isdigit() and 0 <= int(hour) <= 23 and 0 <= int(minute) <= 59):
            raise ValueError("sold_out_time must use HH:MM format")
        return value


class Recommendation(BaseModel):
    service_level_quantile: float = Field(ge=0, le=1)
    dough_balls: int = Field(ge=0)
    dough_grams: int = Field(ge=0)
    reasoning: str


class Forecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_date: date
    prepare_on: date
    horizon_days: int = Field(ge=1)
    unit: Literal["dough_balls"] = "dough_balls"
    p10: float = Field(ge=0)
    p50: float = Field(ge=0)
    p90: float = Field(ge=0)
    model: str
    n_train_days: int = Field(ge=0)
    baseline_p50: float | None = Field(default=None, ge=0)
    recommendation: Recommendation


class VoiceResponse(BaseModel):
    transcript: str
    extracted: DailyRecord | list[DailyRecord]
    warnings: list[str] = Field(default_factory=list)


class RecordsResponse(BaseModel):
    records: list[DailyRecord]


class StatusResponse(BaseModel):
    rows_available: int = Field(ge=0)
    rows_required: int = Field(ge=0)
    ready: bool


class HealthResponse(BaseModel):
    status: Literal["ok", "warming_up", "error"]
    models_loaded: bool
    versions: dict[str, str] = Field(default_factory=dict)
    offline: bool
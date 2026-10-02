"""Environment-backed application settings."""

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


TARGET_COLUMN = os.getenv("TARGET_COLUMN", "pizzas_sold")
HORIZON_DAYS = max(1, _env_int("HORIZON_DAYS", 2))
SERVICE_LEVEL_QUANTILE = _env_float("SERVICE_LEVEL_QUANTILE", 0.8)
BALLS_PER_PIZZA = _env_float("BALLS_PER_PIZZA", 1.0)
GRAMS_PER_BALL = _env_int("GRAMS_PER_BALL", 180)
MIN_TRAIN_ROWS = _env_int("MIN_TRAIN_ROWS", 28)
USE_RECALLED_IN_FIT = _env_bool("USE_RECALLED_IN_FIT", False)
HOLIDAY_COUNTRY = os.getenv("HOLIDAY_COUNTRY", "IT")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
TABPFN_VERSION = os.getenv("TABPFN_VERSION", "default")
TABPFN_TOKEN = os.getenv("TABPFN_TOKEN")
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", "data/private/pizzeria.db"))
RECIPE_FLOUR_PERCENT = os.getenv("RECIPE_FLOUR_PERCENT")
RECIPE_WATER_PERCENT = os.getenv("RECIPE_WATER_PERCENT")
RECIPE_SALT_PERCENT = os.getenv("RECIPE_SALT_PERCENT")
RECIPE_YEAST_PERCENT = os.getenv("RECIPE_YEAST_PERCENT")
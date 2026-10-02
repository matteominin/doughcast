"""Convert demand intervals into dough quantities without inventing text."""

from __future__ import annotations

import math
from typing import Mapping

from dough.config import BALLS_PER_PIZZA, GRAMS_PER_BALL, RECIPE_FLOUR_PERCENT, RECIPE_SALT_PERCENT, RECIPE_WATER_PERCENT, RECIPE_YEAST_PERCENT, SERVICE_LEVEL_QUANTILE
from dough.schemas import Forecast, Recommendation


def _value(value: str | None) -> float | None:
    return float(value) if value is not None else None


def tradeoff_table(p10: float, p50: float, p90: float, balls_per_pizza: float = BALLS_PER_PIZZA) -> list[dict[str, float | int]]:
    quantiles = {0.5: p50, 0.7: p50 + 0.4 * (p90 - p50), 0.8: p50 + 0.6 * (p90 - p50), 0.9: p90}
    return [{"service_level_quantile": q, "dough_balls": math.ceil(max(0, value * balls_per_pizza))} for q, value in quantiles.items()]


def recommendation_from_forecast(forecast: Forecast, *, service_level_quantile: float = SERVICE_LEVEL_QUANTILE, recipe_percentages: Mapping[str, float | None] | None = None) -> Forecast:
    """Return a copy with a rounded, mass-based recommendation attached."""
    if not 0 <= service_level_quantile <= 1:
        raise ValueError("service_level_quantile must be between 0 and 1")
    estimate = {0.1: forecast.p10, 0.5: forecast.p50, 0.9: forecast.p90}.get(service_level_quantile)
    if estimate is None:
        estimate = forecast.p10 + ((forecast.p90 - forecast.p10) * service_level_quantile)
    balls = math.ceil(max(0, estimate * BALLS_PER_PIZZA))
    grams = balls * GRAMS_PER_BALL
    recipe = recipe_percentages or {"flour": _value(RECIPE_FLOUR_PERCENT), "water": _value(RECIPE_WATER_PERCENT), "salt": _value(RECIPE_SALT_PERCENT), "yeast": _value(RECIPE_YEAST_PERCENT)}
    reasoning = f"Per il {service_level_quantile:.0%} di servizio: prepara {balls} panetti ({grams} g), per dopodomani."
    if any(value is not None for value in recipe.values()):
        reasoning += " Le quantita della ricetta sono disponibili nella configurazione."
    return forecast.model_copy(update={"recommendation": Recommendation(service_level_quantile=service_level_quantile, dough_balls=balls, dough_grams=grams, reasoning=reasoning)})
"""Weather hook (Open-Meteo). Off by default.

To turn it on, allow api.open-meteo.com and historical-forecast-api.open-meteo.com in the cloud
environment's network settings and set [weather] enabled = true in config/model.toml.

Backtests must use archived forecasts (historical-forecast-api), never observed weather: observed
conditions are not known at kickoff. The schedule's temp/wind columns are observed and are blocked.
Domes return None. Any error returns None, so weather can never break a report.
Weather stays display-only until the Phase 7 loop shows it improves validation Brier score.
"""
from __future__ import annotations

import pandas as pd
import requests

from .. import config
from .context import stadiums

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
HOURLY = "temperature_2m,precipitation,wind_speed_10m,wind_gusts_10m"


def enabled() -> bool:
    return bool(config.model().get("weather", {}).get("enabled", False))


def kickoff_weather(stadium_id: str, kickoff_utc: pd.Timestamp, roof_pregame: str, historical: bool = False,
                    timeout: int = 15) -> dict | None:
    if not enabled() or roof_pregame == "dome":
        return None
    try:
        st = stadiums().loc[stadium_id]
        day = kickoff_utc.strftime("%Y-%m-%d")
        params = dict(latitude=float(st["lat"]), longitude=float(st["lon"]), hourly=HOURLY, timezone="UTC",
                      start_date=day, end_date=day, wind_speed_unit="mph", temperature_unit="fahrenheit",
                      precipitation_unit="inch")
        r = requests.get(HISTORICAL_FORECAST_URL if historical else FORECAST_URL, params=params, timeout=timeout)
        r.raise_for_status()
        h = r.json()["hourly"]
        times = pd.to_datetime(h["time"], utc=True)
        i = int(abs(times - kickoff_utc.floor("h")).argmin())
        return dict(temp_f=h["temperature_2m"][i], precip_in=h["precipitation"][i], wind_mph=h["wind_speed_10m"][i],
                    gust_mph=h["wind_gusts_10m"][i], source="open-meteo " + ("historical forecast" if historical
                                                                            else "forecast"))
    except Exception:
        return None

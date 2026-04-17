"""
data/world_bank_client.py — World Bank Open Data API client.

Fetches macroeconomic indicators for all 32 World Cup nations and caches
responses locally.  Falls back to hardcoded 2024 estimates when the API is
unavailable (no internet, rate-limit, etc.).

Base URL: https://api.worldbank.org/v2/
Indicators used:
  NY.GDP.PCAP.CD  — GDP per capita (current USD)
  SP.POP.TOTL     — Total population
  SE.XPD.TOTL.GD.ZS — Government expenditure on education (% of GDP)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://api.worldbank.org/v2/"
CACHE_DIR = Path(".cache")
CACHE_TTL_SECONDS = 86_400 * 7  # 7 days

INDICATORS = {
    "gdp_per_capita": "NY.GDP.PCAP.CD",
    "population":     "SP.POP.TOTL",
    "education_pct":  "SE.XPD.TOTL.GD.ZS",
}

# ISO-2 codes for all 32 2026 World Cup nations (projected group)
WC2026_ISO2: dict[str, str] = {
    "Argentina":     "AR",
    "Australia":     "AU",
    "Belgium":       "BE",
    "Brazil":        "BR",
    "Cameroon":      "CM",
    "Canada":        "CA",
    "Croatia":       "HR",
    "Denmark":       "DK",
    "Ecuador":       "EC",
    "England":       "GB",   # World Bank uses GB for UK
    "France":        "FR",
    "Germany":       "DE",
    "Ghana":         "GH",
    "Iran":          "IR",
    "Japan":         "JP",
    "Mexico":        "MX",
    "Morocco":       "MA",
    "Netherlands":   "NL",
    "Nigeria":       "NG",
    "Poland":        "PL",
    "Portugal":      "PT",
    "Qatar":         "QA",
    "Saudi Arabia":  "SA",
    "Senegal":       "SN",
    "Serbia":        "RS",
    "South Korea":   "KR",
    "Spain":         "ES",
    "Switzerland":   "CH",
    "Tunisia":       "TN",
    "Uruguay":       "UY",
    "USA":           "US",
    "Wales":         "GB",   # Wales uses GB at World Bank level
}

# Hardcoded 2024 fallback data (World Bank 2023 estimates)
_FALLBACK_DATA: dict[str, dict[str, float]] = {
    "Argentina":   {"gdp_per_capita": 13_700,  "population": 46_300_000, "education_pct": 5.1},
    "Australia":   {"gdp_per_capita": 65_100,  "population": 26_500_000, "education_pct": 5.5},
    "Belgium":     {"gdp_per_capita": 51_200,  "population": 11_700_000, "education_pct": 6.5},
    "Brazil":      {"gdp_per_capita": 10_300,  "population": 215_300_000,"education_pct": 5.9},
    "Cameroon":    {"gdp_per_capita":  1_620,  "population": 28_500_000, "education_pct": 3.0},
    "Canada":      {"gdp_per_capita": 54_900,  "population": 39_200_000, "education_pct": 5.2},
    "Croatia":     {"gdp_per_capita": 21_600,  "population":  3_900_000, "education_pct": 5.3},
    "Denmark":     {"gdp_per_capita": 68_000,  "population":  5_900_000, "education_pct": 7.0},
    "Ecuador":     {"gdp_per_capita":  6_300,  "population": 18_100_000, "education_pct": 4.8},
    "England":     {"gdp_per_capita": 47_300,  "population": 56_500_000, "education_pct": 5.5},
    "France":      {"gdp_per_capita": 43_700,  "population": 68_100_000, "education_pct": 5.5},
    "Germany":     {"gdp_per_capita": 51_200,  "population": 84_400_000, "education_pct": 4.9},
    "Ghana":       {"gdp_per_capita":  2_380,  "population": 33_500_000, "education_pct": 4.2},
    "Iran":        {"gdp_per_capita":  4_600,  "population": 88_000_000, "education_pct": 3.6},
    "Japan":       {"gdp_per_capita": 39_300,  "population": 124_600_000,"education_pct": 3.2},
    "Mexico":      {"gdp_per_capita": 11_500,  "population": 130_200_000,"education_pct": 4.3},
    "Morocco":     {"gdp_per_capita":  3_800,  "population": 37_700_000, "education_pct": 5.3},
    "Netherlands": {"gdp_per_capita": 58_300,  "population": 17_900_000, "education_pct": 5.3},
    "Nigeria":     {"gdp_per_capita":  2_180,  "population": 218_500_000,"education_pct": 0.6},
    "Poland":      {"gdp_per_capita": 18_000,  "population": 36_700_000, "education_pct": 4.9},
    "Portugal":    {"gdp_per_capita": 24_500,  "population": 10_200_000, "education_pct": 4.9},
    "Qatar":       {"gdp_per_capita": 83_900,  "population":  2_900_000, "education_pct": 3.3},
    "Saudi Arabia":{"gdp_per_capita": 28_200,  "population": 36_400_000, "education_pct": 7.0},
    "Senegal":     {"gdp_per_capita":  1_640,  "population": 17_700_000, "education_pct": 4.9},
    "Serbia":      {"gdp_per_capita":  9_700,  "population":  6_800_000, "education_pct": 3.6},
    "South Korea": {"gdp_per_capita": 33_100,  "population": 51_700_000, "education_pct": 4.9},
    "Spain":       {"gdp_per_capita": 32_000,  "population": 47_500_000, "education_pct": 4.3},
    "Switzerland": {"gdp_per_capita": 92_400,  "population":  8_800_000, "education_pct": 5.5},
    "Tunisia":     {"gdp_per_capita":  3_900,  "population": 12_000_000, "education_pct": 6.8},
    "Uruguay":     {"gdp_per_capita": 17_300,  "population":  3_500_000, "education_pct": 4.9},
    "USA":         {"gdp_per_capita": 80_000,  "population": 335_000_000,"education_pct": 5.0},
    "Wales":       {"gdp_per_capita": 30_000,  "population":  3_200_000, "education_pct": 5.5},
}


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(indicator: str, country: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    safe_country = country.replace(" ", "_")
    return CACHE_DIR / f"worldbank_{indicator}_{safe_country}.json"


def _load_cache(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        age = time.time() - payload.get("_fetched_at", 0)
        if age < CACHE_TTL_SECONDS:
            return payload.get("data")
    except (json.JSONDecodeError, KeyError):
        pass
    return None


def _save_cache(path: Path, data: Any) -> None:
    payload = {"_fetched_at": time.time(), "data": data}
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------

def _fetch_indicator(iso2: str, indicator_code: str) -> Optional[float]:
    """Fetch a single indicator value for one country from the World Bank API."""
    url = (
        f"{BASE_URL}country/{iso2}/indicator/{indicator_code}"
        "?format=json&mrv=1&per_page=1"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        body = resp.json()
        if isinstance(body, list) and len(body) >= 2:
            records = body[1]
            if records and records[0].get("value") is not None:
                return float(records[0]["value"])
    except Exception as exc:  # noqa: BLE001
        logger.debug("World Bank API error for %s/%s: %s", iso2, indicator_code, exc)
    return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def fetch_country_data(country: str) -> dict[str, float]:
    """
    Return macroeconomic indicators for *country*.

    Checks cache first; falls back to live API; falls back to hardcoded data.

    Parameters
    ----------
    country:
        Country name as used in WC2026_ISO2 mapping.

    Returns
    -------
    dict with keys: gdp_per_capita, population, education_pct
    """
    iso2 = WC2026_ISO2.get(country)
    result: dict[str, float] = {}

    for key, indicator_code in INDICATORS.items():
        cache_path = _cache_path(indicator_code, country)
        cached = _load_cache(cache_path)

        if cached is not None:
            result[key] = cached
            continue

        if iso2:
            value = _fetch_indicator(iso2, indicator_code)
            if value is not None:
                result[key] = value
                _save_cache(cache_path, value)
                continue

        # Fallback
        fallback = _FALLBACK_DATA.get(country, {})
        result[key] = fallback.get(key, 0.0)
        logger.debug("Using fallback for %s / %s", country, key)

    return result


def fetch_all_teams() -> dict[str, dict[str, float]]:
    """
    Fetch World Bank data for all 32 WC nations.

    Returns
    -------
    dict mapping country name → indicator dict
    """
    all_data: dict[str, dict[str, float]] = {}
    for country in WC2026_ISO2:
        try:
            all_data[country] = fetch_country_data(country)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch data for %s: %s", country, exc)
            all_data[country] = _FALLBACK_DATA.get(country, {})
    return all_data

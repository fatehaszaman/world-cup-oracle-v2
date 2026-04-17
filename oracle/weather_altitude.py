"""
oracle/weather_altitude.py — 2026 World Cup venue conditions model.

Models how altitude and temperature affect team performance.

2026 World Cup is co-hosted by USA, Canada, and Mexico.
Key altitude venues:
  - Estadio Azteca, Mexico City: 2,240m
  - Estadio Akron, Guadalajara: 1,566m
  - Estadio BBVA, Monterrey: 537m
  - Denver (if applicable): ~1,609m elevation city

Teams from high-altitude nations (Argentina, Bolivia, Colombia, Mexico,
Ecuador) have documented physiological adaptation advantages at altitude.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 2026 WC Venues
# ---------------------------------------------------------------------------

@dataclass
class Venue:
    name:           str
    city:           str
    country:        str
    altitude_m:     float    # metres above sea level
    avg_game_temp_c: float   # average game-time temperature (°C)
    capacity:       int
    timezone:       str


WC2026_VENUES: dict[str, Venue] = {
    # USA venues
    "MetLife Stadium":       Venue("MetLife Stadium",       "East Rutherford NJ", "USA",     3.0,  18.0, 82_500, "America/New_York"),
    "AT&T Stadium":          Venue("AT&T Stadium",          "Arlington TX",       "USA",    183.0,  28.0, 80_000, "America/Chicago"),
    "SoFi Stadium":          Venue("SoFi Stadium",          "Los Angeles CA",     "USA",    116.0,  23.0, 70_200, "America/Los_Angeles"),
    "Levi's Stadium":        Venue("Levi's Stadium",        "Santa Clara CA",     "USA",     18.0,  20.0, 68_500, "America/Los_Angeles"),
    "Arrowhead Stadium":     Venue("Arrowhead Stadium",     "Kansas City MO",     "USA",    296.0,  26.0, 72_000, "America/Chicago"),
    "Rose Bowl":             Venue("Rose Bowl",             "Pasadena CA",        "USA",    236.0,  22.0, 88_565, "America/Los_Angeles"),
    "Hard Rock Stadium":     Venue("Hard Rock Stadium",     "Miami Gardens FL",   "USA",      3.0,  30.0, 65_000, "America/New_York"),
    "Lincoln Financial Field":Venue("Lincoln Financial Field","Philadelphia PA",  "USA",     12.0,  22.0, 69_328, "America/New_York"),
    "Gillette Stadium":      Venue("Gillette Stadium",      "Foxborough MA",      "USA",     24.0,  19.0, 65_878, "America/New_York"),
    "Seattle Stadium":       Venue("Lumen Field",           "Seattle WA",         "USA",      4.0,  18.0, 68_740, "America/Los_Angeles"),
    # Canada venues
    "BC Place":              Venue("BC Place",              "Vancouver BC",       "Canada",   4.0,  18.0, 54_500, "America/Vancouver"),
    "BMO Field":             Venue("BMO Field",             "Toronto ON",         "Canada",   76.0, 22.0, 45_000, "America/Toronto"),
    # Mexico venues
    "Estadio Azteca":        Venue("Estadio Azteca",        "Mexico City",        "Mexico", 2_240.0, 17.0, 87_523, "America/Mexico_City"),
    "Estadio Akron":         Venue("Estadio Akron",         "Guadalajara",        "Mexico", 1_566.0, 22.0, 46_232, "America/Mexico_City"),
    "Estadio BBVA":          Venue("Estadio BBVA",          "Monterrey",          "Mexico",   537.0, 30.0, 53_500, "America/Monterrey"),
    "Estadio Ciudad de Monterrey": Venue("Estadio Ciudad de Monterrey","Monterrey","Mexico",537.0, 30.0, 53_500, "America/Monterrey"),
}

# ---------------------------------------------------------------------------
# Altitude adaptation profiles
# ---------------------------------------------------------------------------

# Teams that regularly train/play at altitude: 0–1 adaptation score
# 1.0 = fully adapted (train above 1500m); 0.0 = sea-level only
_ALTITUDE_ADAPTATION: dict[str, float] = {
    "Bolivia":       1.00,   # La Paz 3,640m
    "Colombia":      0.75,   # Bogotá 2,600m
    "Ecuador":       0.80,   # Quito 2,850m
    "Mexico":        0.85,   # Mexico City 2,240m; Guadalajara 1,566m
    "Peru":          0.65,
    "Argentina":     0.45,   # Buenos Aires sea-level; some high-alt training
    "Chile":         0.40,
    "USA":           0.15,   # Denver at altitude but most teams based sea-level
    "Brazil":        0.05,
    "England":       0.02,
    "France":        0.02,
    "Germany":       0.02,
    "Spain":         0.03,
    "Portugal":      0.02,
    "Netherlands":   0.01,
    "Belgium":       0.01,
    "Japan":         0.05,
    "South Korea":   0.05,
    "Morocco":       0.08,   # Atlas Mountains training
    "Croatia":       0.03,
    "Denmark":       0.01,
    "Switzerland":   0.20,   # Alpine training
    "Poland":        0.02,
    "Serbia":        0.05,
    "Uruguay":       0.05,
    "Canada":        0.05,
    "Senegal":       0.03,
    "Nigeria":       0.03,
    "Ghana":         0.03,
    "Cameroon":      0.03,
    "Iran":          0.10,
    "Australia":     0.02,
    "Wales":         0.02,
    "Tunisia":       0.03,
    "Saudi Arabia":  0.03,
    "Qatar":         0.02,
    "Costa Rica":    0.08,
}

# Preferred temperature range per team (based on domestic league climate)
_PREFERRED_TEMP_RANGE: dict[str, tuple[float, float]] = {
    # (min_comfortable, max_comfortable) in °C
    "Argentina":     (15.0, 28.0),
    "Brazil":        (20.0, 35.0),
    "Mexico":        (18.0, 33.0),
    "Spain":         (12.0, 30.0),
    "Portugal":      (10.0, 28.0),
    "France":        (8.0,  24.0),
    "Germany":       (5.0,  22.0),
    "England":       (5.0,  20.0),
    "Netherlands":   (5.0,  20.0),
    "Belgium":       (5.0,  20.0),
    "Denmark":       (3.0,  18.0),
    "Switzerland":   (3.0,  20.0),
    "Norway":        (0.0,  15.0),
    "Sweden":        (0.0,  15.0),
    "Japan":         (12.0, 30.0),
    "South Korea":   (10.0, 28.0),
    "USA":           (10.0, 30.0),
    "Canada":        (5.0,  22.0),
    "Morocco":       (15.0, 33.0),
    "Senegal":       (22.0, 38.0),
    "Nigeria":       (22.0, 38.0),
    "Ghana":         (22.0, 38.0),
    "Cameroon":      (20.0, 36.0),
    "Saudi Arabia":  (20.0, 38.0),
    "Iran":          (12.0, 32.0),
    "Qatar":         (20.0, 38.0),
    "Ecuador":       (14.0, 26.0),
    "Uruguay":       (12.0, 28.0),
    "Croatia":       (8.0,  26.0),
    "Serbia":        (5.0,  24.0),
    "Poland":        (3.0,  20.0),
    "Tunisia":       (16.0, 35.0),
    "Australia":     (15.0, 30.0),
    "Wales":         (4.0,  20.0),
    "Costa Rica":    (18.0, 32.0),
}

# Approximate lat/lon for travel fatigue calculation (city centroids)
_CITY_COORDS: dict[str, tuple[float, float]] = {
    "East Rutherford NJ": (40.81, -74.08),
    "Arlington TX":       (32.75, -97.09),
    "Los Angeles CA":     (34.05, -118.24),
    "Santa Clara CA":     (37.35, -121.95),
    "Kansas City MO":     (39.10, -94.58),
    "Pasadena CA":        (34.15, -118.14),
    "Miami Gardens FL":   (25.96, -80.24),
    "Philadelphia PA":    (39.95, -75.16),
    "Foxborough MA":      (42.09, -71.26),
    "Seattle WA":         (47.60, -122.33),
    "Vancouver BC":       (49.25, -123.10),
    "Toronto ON":         (43.65, -79.38),
    "Mexico City":        (19.43, -99.13),
    "Guadalajara":        (20.67, -103.35),
    "Monterrey":          (25.67, -100.31),
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def altitude_disadvantage(team: str, venue_altitude_m: float) -> float:
    """
    Compute altitude performance penalty for a team at a given venue.

    Teams with low altitude adaptation playing at high altitude
    (>1500m) suffer aerobic capacity reduction of 5–20%.

    Parameters
    ----------
    team:
        Country name.
    venue_altitude_m:
        Venue altitude in metres above sea level.

    Returns
    -------
    float in [-1, 0]: negative = disadvantage (multiply strength by 1 + this).
    0.0 means no penalty; -0.12 means 12% strength reduction.
    """
    if venue_altitude_m < 500:
        return 0.0   # sea-level: no effect

    # Physiological penalty increases with altitude
    # Roughly 1% per 100m above 1000m for unadapted teams
    altitude_factor = max(0.0, (venue_altitude_m - 1000.0) / 100.0) * 0.01

    adaptation = _ALTITUDE_ADAPTATION.get(team, 0.05)
    # Adapted teams suffer much less
    penalty = altitude_factor * (1.0 - adaptation)
    return round(-min(penalty, 0.20), 4)   # cap at -20%


def temperature_factor(team: str, venue_temp_celsius: float) -> float:
    """
    Compute temperature performance modifier for a team at a given venue.

    Teams playing outside their comfortable temperature range perform worse.

    Parameters
    ----------
    team:
        Country name.
    venue_temp_celsius:
        Expected game-time temperature at the venue (°C).

    Returns
    -------
    float in [-0.10, 0.05]: negative = disadvantage, slight positive = comfort.
    """
    lo, hi = _PREFERRED_TEMP_RANGE.get(team, (10.0, 26.0))

    if lo <= venue_temp_celsius <= hi:
        return 0.02   # slight positive for comfortable conditions

    if venue_temp_celsius < lo:
        gap = lo - venue_temp_celsius
    else:
        gap = venue_temp_celsius - hi

    penalty = min(gap * 0.008, 0.10)
    return round(-penalty, 4)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km between two coordinates."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def travel_fatigue(
    team: str,
    prev_venue_city: str,
    curr_venue_city: str,
    days_between: int = 4,
) -> float:
    """
    Estimate travel fatigue penalty from one venue city to the next.

    Parameters
    ----------
    team:
        Country name (unused currently but reserved for future home-region logic).
    prev_venue_city:
        City name of the previous match venue.
    curr_venue_city:
        City name of the next match venue.
    days_between:
        Days between the two matches. Default 4 (typical WC schedule).

    Returns
    -------
    float in [-0.05, 0.0]: negative = fatigue penalty.
    """
    prev_coords = _CITY_COORDS.get(prev_venue_city)
    curr_coords = _CITY_COORDS.get(curr_venue_city)

    if prev_coords is None or curr_coords is None:
        return 0.0

    dist_km = _haversine_km(*prev_coords, *curr_coords)

    # Penalty: 0.5% per 1000km, scaled down by recovery days
    raw_penalty = (dist_km / 1000.0) * 0.005
    recovery_factor = max(0.2, 1.0 - days_between * 0.15)
    penalty = raw_penalty * recovery_factor

    return round(-min(penalty, 0.05), 4)


def venue_adjustment(
    team: str,
    venue_name: str,
    prev_venue_name: Optional[str] = None,
    days_between: int = 4,
) -> dict[str, float]:
    """
    Compute all venue-related adjustments for a team at a given venue.

    Parameters
    ----------
    team:
        Country name.
    venue_name:
        Key in WC2026_VENUES.
    prev_venue_name:
        Previous match venue key (for travel fatigue). None = no travel fatigue.
    days_between:
        Days since previous match.

    Returns
    -------
    dict with keys: altitude, temperature, travel, total
    """
    venue = WC2026_VENUES.get(venue_name)
    if venue is None:
        logger.warning("Unknown venue: %s", venue_name)
        return {"altitude": 0.0, "temperature": 0.0, "travel": 0.0, "total": 0.0}

    alt_adj  = altitude_disadvantage(team, venue.altitude_m)
    temp_adj = temperature_factor(team, venue.avg_game_temp_c)

    travel_adj = 0.0
    if prev_venue_name and prev_venue_name in WC2026_VENUES:
        prev_city = WC2026_VENUES[prev_venue_name].city
        curr_city = venue.city
        travel_adj = travel_fatigue(team, prev_city, curr_city, days_between)

    total = alt_adj + temp_adj + travel_adj

    return {
        "altitude":    alt_adj,
        "temperature": temp_adj,
        "travel":      travel_adj,
        "total":       round(total, 4),
    }

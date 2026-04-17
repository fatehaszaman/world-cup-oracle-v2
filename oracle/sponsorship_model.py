"""
oracle/sponsorship_model.py — Commercial signal and sponsorship valuation model.

BUSINESS SUMMARY
----------------
Wealthier football federations invest more in youth academies, coaching
infrastructure, and player development — creating a talent pipeline that
shows up in World Cup results years later. This module uses sponsorship
revenue as a measurable proxy for that investment cycle. A federation
earning €200M/year from shirt deals and broadcast rights simply has more
money to develop the next generation than one earning €20M/year.

DEVELOPER NOTES
---------------
Data sources: Public reporting from Forbes, SportsPro Media, GlobalData
(2024–25 estimates). All figures are annual EUR millions unless noted.

Normalization: Each sub-signal is linearly normalised against a ceiling
constant defined in config.py. The composite score is an equally-weighted
blend of five normalized dimensions.

Complexity: O(1) per team lookup. O(T) for compare_all().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import (
    SHIRT_DEAL_CEILING_EUR_M,
    KIT_DEAL_CEILING_EUR_M,
    FED_REVENUE_CEILING_EUR_M,
    SOCIAL_FOLLOWERS_CEILING_M,
    FANBASE_INDEX_CEILING,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SponsorshipProfile:
    """
    Full commercial data record for one national team.

    Attributes
    ----------
    shirt_sponsor : str            Primary shirt sponsor brand name.
    shirt_deal_eur_m : float       Annual shirt sponsorship value (EUR millions).
    kit_manufacturer : str         Kit maker (e.g., Nike, Adidas, Puma).
    kit_deal_eur_m : float         Annual kit manufacturing deal (EUR millions).
    fed_commercial_revenue_eur_m : float
        Total federation annual commercial revenue including broadcast,
        naming rights, and licensing (EUR millions).
    social_followers_m : float     Combined social media following (millions).
    global_fanbase_index : float   Proprietary index 0–10 from YouGov/Nielsen data.
    """
    shirt_sponsor: str
    shirt_deal_eur_m: float
    kit_manufacturer: str
    kit_deal_eur_m: float
    fed_commercial_revenue_eur_m: float
    social_followers_m: float
    global_fanbase_index: float


# ---------------------------------------------------------------------------
# Hardcoded commercial data — 2025/26 estimates
# Sources: Forbes, SportsPro, GlobalData, public federation reports
# ---------------------------------------------------------------------------
SPONSORSHIP_DATA: dict[str, SponsorshipProfile] = {
    "England": SponsorshipProfile(
        shirt_sponsor="IHG Hotels & Resorts",
        shirt_deal_eur_m=55.0,
        kit_manufacturer="Nike",
        kit_deal_eur_m=95.0,
        fed_commercial_revenue_eur_m=230.0,
        social_followers_m=120.0,
        global_fanbase_index=9.2,
    ),
    "France": SponsorshipProfile(
        shirt_sponsor="Bpifrance",
        shirt_deal_eur_m=48.0,
        kit_manufacturer="Nike",
        kit_deal_eur_m=52.0,
        fed_commercial_revenue_eur_m=195.0,
        social_followers_m=100.0,
        global_fanbase_index=8.8,
    ),
    "Brazil": SponsorshipProfile(
        shirt_sponsor="Guaraná Antarctica",
        shirt_deal_eur_m=42.0,
        kit_manufacturer="Nike",
        kit_deal_eur_m=85.0,
        fed_commercial_revenue_eur_m=185.0,
        social_followers_m=145.0,
        global_fanbase_index=9.5,
    ),
    "Germany": SponsorshipProfile(
        shirt_sponsor="Deutsche Telekom",
        shirt_deal_eur_m=45.0,
        kit_manufacturer="Adidas",
        kit_deal_eur_m=65.0,
        fed_commercial_revenue_eur_m=175.0,
        social_followers_m=95.0,
        global_fanbase_index=8.5,
    ),
    "Argentina": SponsorshipProfile(
        shirt_sponsor="Mercado Libre",
        shirt_deal_eur_m=35.0,
        kit_manufacturer="Adidas",
        kit_deal_eur_m=55.0,
        fed_commercial_revenue_eur_m=140.0,
        social_followers_m=110.0,
        global_fanbase_index=9.0,
    ),
    "Spain": SponsorshipProfile(
        shirt_sponsor="Rakuten",
        shirt_deal_eur_m=30.0,
        kit_manufacturer="Adidas",
        kit_deal_eur_m=58.0,
        fed_commercial_revenue_eur_m=160.0,
        social_followers_m=90.0,
        global_fanbase_index=8.6,
    ),
    "Portugal": SponsorshipProfile(
        shirt_sponsor="Hankook Tire",
        shirt_deal_eur_m=25.0,
        kit_manufacturer="Nike",
        kit_deal_eur_m=40.0,
        fed_commercial_revenue_eur_m=110.0,
        social_followers_m=85.0,
        global_fanbase_index=8.0,
    ),
    "Netherlands": SponsorshipProfile(
        shirt_sponsor="ING",
        shirt_deal_eur_m=28.0,
        kit_manufacturer="Nike",
        kit_deal_eur_m=38.0,
        fed_commercial_revenue_eur_m=120.0,
        social_followers_m=55.0,
        global_fanbase_index=7.4,
    ),
    "Belgium": SponsorshipProfile(
        shirt_sponsor="Visit Brussels",
        shirt_deal_eur_m=18.0,
        kit_manufacturer="Adidas",
        kit_deal_eur_m=30.0,
        fed_commercial_revenue_eur_m=90.0,
        social_followers_m=42.0,
        global_fanbase_index=6.8,
    ),
    "Italy": SponsorshipProfile(
        shirt_sponsor="Enel",
        shirt_deal_eur_m=32.0,
        kit_manufacturer="Puma",
        kit_deal_eur_m=35.0,
        fed_commercial_revenue_eur_m=130.0,
        social_followers_m=62.0,
        global_fanbase_index=7.8,
    ),
    "Croatia": SponsorshipProfile(
        shirt_sponsor="Croatia Airlines",
        shirt_deal_eur_m=8.0,
        kit_manufacturer="Nike",
        kit_deal_eur_m=15.0,
        fed_commercial_revenue_eur_m=40.0,
        social_followers_m=18.0,
        global_fanbase_index=5.2,
    ),
    "Uruguay": SponsorshipProfile(
        shirt_sponsor="Tenfield",
        shirt_deal_eur_m=7.0,
        kit_manufacturer="Puma",
        kit_deal_eur_m=12.0,
        fed_commercial_revenue_eur_m=38.0,
        social_followers_m=15.0,
        global_fanbase_index=5.0,
    ),
    "Mexico": SponsorshipProfile(
        shirt_sponsor="Telcel",
        shirt_deal_eur_m=22.0,
        kit_manufacturer="Adidas",
        kit_deal_eur_m=45.0,
        fed_commercial_revenue_eur_m=120.0,
        social_followers_m=60.0,
        global_fanbase_index=7.2,
    ),
    "Colombia": SponsorshipProfile(
        shirt_sponsor="Allianz",
        shirt_deal_eur_m=12.0,
        kit_manufacturer="Adidas",
        kit_deal_eur_m=22.0,
        fed_commercial_revenue_eur_m=60.0,
        social_followers_m=28.0,
        global_fanbase_index=5.8,
    ),
    "Senegal": SponsorshipProfile(
        shirt_sponsor="Orange Sénégal",
        shirt_deal_eur_m=4.0,
        kit_manufacturer="Puma",
        kit_deal_eur_m=8.0,
        fed_commercial_revenue_eur_m=22.0,
        social_followers_m=12.0,
        global_fanbase_index=4.5,
    ),
    "Morocco": SponsorshipProfile(
        shirt_sponsor="BMCE Bank",
        shirt_deal_eur_m=6.0,
        kit_manufacturer="Puma",
        kit_deal_eur_m=10.0,
        fed_commercial_revenue_eur_m=28.0,
        social_followers_m=20.0,
        global_fanbase_index=5.0,
    ),
    "United States": SponsorshipProfile(
        shirt_sponsor="Volkswagen",
        shirt_deal_eur_m=20.0,
        kit_manufacturer="Nike",
        kit_deal_eur_m=55.0,
        fed_commercial_revenue_eur_m=140.0,
        social_followers_m=45.0,
        global_fanbase_index=6.0,
    ),
    "Japan": SponsorshipProfile(
        shirt_sponsor="DAZN",
        shirt_deal_eur_m=15.0,
        kit_manufacturer="Adidas",
        kit_deal_eur_m=28.0,
        fed_commercial_revenue_eur_m=70.0,
        social_followers_m=32.0,
        global_fanbase_index=6.2,
    ),
    "South Korea": SponsorshipProfile(
        shirt_sponsor="Hyundai",
        shirt_deal_eur_m=16.0,
        kit_manufacturer="Nike",
        kit_deal_eur_m=25.0,
        fed_commercial_revenue_eur_m=65.0,
        social_followers_m=30.0,
        global_fanbase_index=6.0,
    ),
    "Switzerland": SponsorshipProfile(
        shirt_sponsor="Helvetia Insurance",
        shirt_deal_eur_m=12.0,
        kit_manufacturer="Puma",
        kit_deal_eur_m=18.0,
        fed_commercial_revenue_eur_m=50.0,
        social_followers_m=14.0,
        global_fanbase_index=5.2,
    ),
    "Denmark": SponsorshipProfile(
        shirt_sponsor="Hummel",
        shirt_deal_eur_m=10.0,
        kit_manufacturer="Hummel",
        kit_deal_eur_m=14.0,
        fed_commercial_revenue_eur_m=45.0,
        social_followers_m=12.0,
        global_fanbase_index=5.0,
    ),
    "Austria": SponsorshipProfile(
        shirt_sponsor="Erste Group",
        shirt_deal_eur_m=8.0,
        kit_manufacturer="Puma",
        kit_deal_eur_m=12.0,
        fed_commercial_revenue_eur_m=38.0,
        social_followers_m=10.0,
        global_fanbase_index=4.5,
    ),
    "Poland": SponsorshipProfile(
        shirt_sponsor="PKN Orlen",
        shirt_deal_eur_m=10.0,
        kit_manufacturer="Nike",
        kit_deal_eur_m=18.0,
        fed_commercial_revenue_eur_m=48.0,
        social_followers_m=16.0,
        global_fanbase_index=5.2,
    ),
    "Serbia": SponsorshipProfile(
        shirt_sponsor="Telekom Srbija",
        shirt_deal_eur_m=5.0,
        kit_manufacturer="Puma",
        kit_deal_eur_m=10.0,
        fed_commercial_revenue_eur_m=28.0,
        social_followers_m=10.0,
        global_fanbase_index=4.3,
    ),
    "Ecuador": SponsorshipProfile(
        shirt_sponsor="Marathon Sports",
        shirt_deal_eur_m=3.0,
        kit_manufacturer="Marathon",
        kit_deal_eur_m=5.0,
        fed_commercial_revenue_eur_m=18.0,
        social_followers_m=8.0,
        global_fanbase_index=3.8,
    ),
    "Canada": SponsorshipProfile(
        shirt_sponsor="Sprinklr",
        shirt_deal_eur_m=5.0,
        kit_manufacturer="Nike",
        kit_deal_eur_m=12.0,
        fed_commercial_revenue_eur_m=30.0,
        social_followers_m=10.0,
        global_fanbase_index=4.0,
    ),
    "Australia": SponsorshipProfile(
        shirt_sponsor="Subway",
        shirt_deal_eur_m=4.0,
        kit_manufacturer="Nike",
        kit_deal_eur_m=10.0,
        fed_commercial_revenue_eur_m=28.0,
        social_followers_m=9.0,
        global_fanbase_index=3.8,
    ),
    "Nigeria": SponsorshipProfile(
        shirt_sponsor="Zenith Bank",
        shirt_deal_eur_m=5.0,
        kit_manufacturer="Nike",
        kit_deal_eur_m=14.0,
        fed_commercial_revenue_eur_m=32.0,
        social_followers_m=18.0,
        global_fanbase_index=5.5,
    ),
    "Ivory Coast": SponsorshipProfile(
        shirt_sponsor="MTN",
        shirt_deal_eur_m=3.5,
        kit_manufacturer="Puma",
        kit_deal_eur_m=8.0,
        fed_commercial_revenue_eur_m=22.0,
        social_followers_m=10.0,
        global_fanbase_index=4.2,
    ),
    "Cameroon": SponsorshipProfile(
        shirt_sponsor="MTN Cameroon",
        shirt_deal_eur_m=2.5,
        kit_manufacturer="One All Sports",
        kit_deal_eur_m=4.0,
        fed_commercial_revenue_eur_m=15.0,
        social_followers_m=8.0,
        global_fanbase_index=3.8,
    ),
    "Saudi Arabia": SponsorshipProfile(
        shirt_sponsor="NEOM",
        shirt_deal_eur_m=18.0,
        kit_manufacturer="Nike",
        kit_deal_eur_m=20.0,
        fed_commercial_revenue_eur_m=65.0,
        social_followers_m=12.0,
        global_fanbase_index=4.5,
    ),
    "Iran": SponsorshipProfile(
        shirt_sponsor="Mahan Air",
        shirt_deal_eur_m=2.0,
        kit_manufacturer="Uhlsport",
        kit_deal_eur_m=3.0,
        fed_commercial_revenue_eur_m=12.0,
        social_followers_m=6.0,
        global_fanbase_index=3.2,
    ),
}


class SponsorshipValuator:
    """
    Computes normalised commercial scores and talent-concentration proxies
    for all 32 2026 World Cup teams.

    The commercial score is a talent-investment proxy: federations with high
    commercial revenue invest more in youth development, leading to better
    long-term squad quality. This correlation is imperfect (small nations
    can punch above weight) but statistically significant at tournament level.

    Methods
    -------
    get_commercial_score(team)              → float [0, 1]
    get_talent_concentration_proxy(team)    → float [0, 1]
    compare_teams(team_a, team_b)           → dict
    compare_all()                           → list[dict]
    """

    def __init__(self) -> None:
        self._db = SPONSORSHIP_DATA

    def _get(self, team: str) -> SponsorshipProfile | None:
        p = self._db.get(team)
        if p is None:
            logger.warning("No sponsorship data for '%s'.", team)
        return p

    def get_commercial_score(self, team: str) -> float:
        """
        Composite normalised commercial score in [0, 1].

        Five sub-signals, equally weighted (0.20 each):
          1. Shirt deal value / SHIRT_DEAL_CEILING
          2. Kit deal value   / KIT_DEAL_CEILING
          3. Fed revenue      / FED_REVENUE_CEILING
          4. Social following / SOCIAL_FOLLOWERS_CEILING
          5. Fanbase index    / FANBASE_INDEX_CEILING

        Parameters
        ----------
        team : str

        Returns
        -------
        float  Commercial score in [0, 1].
        """
        p = self._get(team)
        if p is None:
            return 0.25

        shirt   = min(p.shirt_deal_eur_m       / SHIRT_DEAL_CEILING_EUR_M,   1.0)
        kit     = min(p.kit_deal_eur_m          / KIT_DEAL_CEILING_EUR_M,     1.0)
        revenue = min(p.fed_commercial_revenue_eur_m / FED_REVENUE_CEILING_EUR_M, 1.0)
        social  = min(p.social_followers_m      / SOCIAL_FOLLOWERS_CEILING_M, 1.0)
        fanbase = min(p.global_fanbase_index    / FANBASE_INDEX_CEILING,      1.0)

        score = (shirt + kit + revenue + social + fanbase) / 5.0
        return round(float(score), 6)

    def get_talent_concentration_proxy(self, team: str) -> float:
        """
        Estimate the talent pipeline strength implied by commercial investment.

        Theory: Higher commercial revenue → larger federation budget →
        more invested in youth academies, coaching infrastructure, and
        elite player development programs. This creates a feedback loop
        where commercially successful federations produce better players.

        The proxy is a dampened version of the commercial score (square-root
        transformation) to reflect diminishing returns beyond a threshold.

        Parameters
        ----------
        team : str

        Returns
        -------
        float  Talent concentration proxy in [0, 1].
        """
        raw = self.get_commercial_score(team)
        # Square-root dampening: high earners don't get proportionally better talent
        return round(float(raw ** 0.65), 6)

    def compare_teams(self, team_a: str, team_b: str) -> dict:
        """
        Side-by-side commercial comparison between two teams.

        Parameters
        ----------
        team_a, team_b : str

        Returns
        -------
        dict with keys: team_a, team_b, comparison (dict of signal → {a, b, winner}),
             commercial_score_a, commercial_score_b, verdict.
        """
        pa = self._get(team_a)
        pb = self._get(team_b)
        score_a = self.get_commercial_score(team_a)
        score_b = self.get_commercial_score(team_b)

        comparison: dict[str, dict] = {}
        if pa and pb:
            fields = [
                ("shirt_deal_eur_m",            "Shirt deal (EUR M)"),
                ("kit_deal_eur_m",               "Kit deal (EUR M)"),
                ("fed_commercial_revenue_eur_m", "Federation revenue (EUR M)"),
                ("social_followers_m",           "Social followers (M)"),
                ("global_fanbase_index",         "Fanbase index (0–10)"),
            ]
            for attr, label in fields:
                va = getattr(pa, attr)
                vb = getattr(pb, attr)
                comparison[label] = {
                    team_a: va,
                    team_b: vb,
                    "winner": team_a if va > vb else (team_b if vb > va else "tie"),
                }

        verdict = (
            f"{team_a} leads commercially ({score_a:.3f} vs {score_b:.3f})"
            if score_a > score_b
            else f"{team_b} leads commercially ({score_b:.3f} vs {score_a:.3f})"
            if score_b > score_a
            else "Commercial parity"
        )

        return {
            "team_a":              team_a,
            "team_b":              team_b,
            "comparison":          comparison,
            "commercial_score_a":  score_a,
            "commercial_score_b":  score_b,
            "verdict":             verdict,
        }

    def compare_all(self) -> list[dict]:
        """
        Return all teams ranked by commercial score descending.

        Returns
        -------
        list[dict]  Each entry: {team, commercial_score, talent_proxy, ...profile fields}
        """
        results = []
        for team, profile in self._db.items():
            results.append({
                "team":             team,
                "commercial_score": self.get_commercial_score(team),
                "talent_proxy":     self.get_talent_concentration_proxy(team),
                "shirt_sponsor":    profile.shirt_sponsor,
                "kit_manufacturer": profile.kit_manufacturer,
                "fed_revenue_eur_m": profile.fed_commercial_revenue_eur_m,
            })
        return sorted(results, key=lambda x: x["commercial_score"], reverse=True)

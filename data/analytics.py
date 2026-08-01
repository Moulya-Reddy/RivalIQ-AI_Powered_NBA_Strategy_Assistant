"""
Derives team analytics from raw game results, since the free API tier
doesn't include pre-built season averages or standings.

This is the "own the data pipeline" piece: given a list of finished games
for a team, compute win/loss record, points scored/allowed, margin, home
vs away splits, and a simple recent-form trend (last N games).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import statistics


@dataclass
class TeamGameResult:
    game_id: int
    date: str
    is_home: bool
    team_score: int
    opp_score: int
    opp_abbreviation: str

    @property
    def won(self) -> bool:
        return self.team_score > self.opp_score

    @property
    def margin(self) -> int:
        return self.team_score - self.opp_score


@dataclass
class TeamProfile:
    team_id: int
    abbreviation: str
    full_name: str
    games: List[TeamGameResult] = field(default_factory=list)

    @property
    def wins(self) -> int:
        return sum(1 for g in self.games if g.won)

    @property
    def losses(self) -> int:
        return sum(1 for g in self.games if not g.won)

    @property
    def win_pct(self) -> float:
        return round(self.wins / len(self.games), 3) if self.games else 0.0

    @property
    def avg_points_scored(self) -> float:
        return round(statistics.mean(g.team_score for g in self.games), 1) if self.games else 0.0

    @property
    def avg_points_allowed(self) -> float:
        return round(statistics.mean(g.opp_score for g in self.games), 1) if self.games else 0.0

    @property
    def avg_margin(self) -> float:
        return round(statistics.mean(g.margin for g in self.games), 1) if self.games else 0.0

    @property
    def home_record(self) -> str:
        home_games = [g for g in self.games if g.is_home]
        w = sum(1 for g in home_games if g.won)
        return f"{w}-{len(home_games) - w}"

    @property
    def away_record(self) -> str:
        away_games = [g for g in self.games if not g.is_home]
        w = sum(1 for g in away_games if g.won)
        return f"{w}-{len(away_games) - w}"

    def recent_form(self, n: int = 5) -> str:
        """W/L string for the last n games, most recent last."""
        recent = sorted(self.games, key=lambda g: g.date)[-n:]
        return "".join("W" if g.won else "L" for g in recent)

    def recent_form_trend(self, n: int = 5) -> float:
        """Average margin over the last n games - a simple momentum signal."""
        recent = sorted(self.games, key=lambda g: g.date)[-n:]
        return round(statistics.mean(g.margin for g in recent), 1) if recent else 0.0

    def to_stats_dict(self) -> Dict:
        return {
            "team": self.full_name,
            "abbreviation": self.abbreviation,
            "games_played": len(self.games),
            "record": f"{self.wins}-{self.losses}",
            "win_pct": self.win_pct,
            "avg_points_scored": self.avg_points_scored,
            "avg_points_allowed": self.avg_points_allowed,
            "avg_margin": self.avg_margin,
            "home_record": self.home_record,
            "away_record": self.away_record,
            "recent_form_last_5": self.recent_form(5),
            "recent_form_margin_last_5": self.recent_form_trend(5),
        }


def build_team_profile(team: dict, games: List[dict]) -> TeamProfile:
    """team: a balldontlie team dict. games: raw game dicts involving this team."""
    profile = TeamProfile(
        team_id=team["id"], abbreviation=team["abbreviation"], full_name=team["full_name"]
    )
    for g in games:
        if g["status"] != "Final":
            continue
        is_home = g["home_team"]["id"] == team["id"]
        if is_home:
            team_score, opp_score = g["home_team_score"], g["visitor_team_score"]
            opp_abbr = g["visitor_team"]["abbreviation"]
        else:
            team_score, opp_score = g["visitor_team_score"], g["home_team_score"]
            opp_abbr = g["home_team"]["abbreviation"]
        profile.games.append(
            TeamGameResult(
                game_id=g["id"],
                date=g["date"],
                is_home=is_home,
                team_score=team_score,
                opp_score=opp_score,
                opp_abbreviation=opp_abbr,
            )
        )
    return profile

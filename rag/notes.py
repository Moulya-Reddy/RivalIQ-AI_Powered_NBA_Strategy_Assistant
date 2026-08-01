"""
Converts derived team analytics into retrievable text "scouting notes".

This is deliberately structured rather than vector-only: each note carries
its source stats dict alongside the text, so the RAG faithfulness eval
(eval/faithfulness.py) can check generated claims against ground truth
numbers, not just against retrieved text.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

from data.analytics import TeamProfile


@dataclass
class ScoutingNote:
    team_abbreviation: str
    text: str
    stats: Dict


def build_scouting_note(profile: TeamProfile) -> ScoutingNote:
    s = profile.to_stats_dict()
    form_desc = {
        "hot": s["recent_form_margin_last_5"] > 5,
        "cold": s["recent_form_margin_last_5"] < -5,
    }
    form_label = "trending hot" if form_desc["hot"] else "trending cold" if form_desc["cold"] else "steady"

    text = (
        f"{s['team']} ({s['abbreviation']}) is {s['record']} ({s['win_pct']:.1%} win rate) "
        f"over the last {s['games_played']} games. They average {s['avg_points_scored']} points "
        f"scored and {s['avg_points_allowed']} allowed per game, for an average margin of "
        f"{s['avg_margin']:+.1f}. Home record: {s['home_record']}. Away record: {s['away_record']}. "
        f"Over their last 5 games they went {s['recent_form_last_5']} "
        f"(avg margin {s['recent_form_margin_last_5']:+.1f}), {form_label}."
    )
    return ScoutingNote(team_abbreviation=s["abbreviation"], text=text, stats=s)


def build_corpus(profiles: List[TeamProfile]) -> List[ScoutingNote]:
    return [build_scouting_note(p) for p in profiles]

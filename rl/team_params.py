"""
Maps real, derived team stats (from data/analytics.py) onto the skill
parameters the RL environment needs. This is the explicit, documented
bridge between "real retrieved data" and "simulated decision environment"
- kept in its own module so the estimation logic is easy to inspect,
question, and improve (a good thing to walk through in an interview).
"""

from __future__ import annotations
from data.analytics import TeamProfile
from rl.environment import TeamSkill

LEAGUE_AVG_PTS_PER_GAME = 113.0  # rough modern-NBA baseline for normalization


def derive_team_skill(profile: TeamProfile) -> TeamSkill:
    stats = profile.to_stats_dict()

    offensive_index = stats["avg_points_scored"] / LEAGUE_AVG_PTS_PER_GAME
    defensive_index = stats["avg_points_allowed"] / LEAGUE_AVG_PTS_PER_GAME

    # Baseline shot success rates (roughly league-average), scaled by how
    # much better/worse this team's scoring efficiency is than league average.
    three_pt_success = min(0.9, 0.36 * offensive_index)
    two_pt_success = min(0.9, 0.50 * offensive_index)

    # Recent-form margin maps to a hot/cold multiplier, clipped to a
    # reasonable range so a hot streak can't make the simulation degenerate.
    hot_factor = 1.0 + max(-0.15, min(0.15, stats["recent_form_margin_last_5"] / 100))

    return TeamSkill(
        three_pt_success=round(three_pt_success, 3),
        two_pt_success=round(two_pt_success, 3),
        hot_factor=round(hot_factor, 3),
        opp_three_pt_allowed=round(min(0.9, 0.36 * defensive_index), 3),
        opp_two_pt_allowed=round(min(0.9, 0.50 * defensive_index), 3),
    )


def derive_opponent_skill(opponent_profile: TeamProfile) -> TeamSkill:
    """The opponent's own offensive skill is what the primary team's
    defense has to face - reuse the same derivation, symmetric treatment."""
    return derive_team_skill(opponent_profile)

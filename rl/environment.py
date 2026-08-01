"""
A simplified "clutch possession" decision environment: given a score
differential and possessions remaining, an agent must choose a shot-
selection/tactical strategy each possession to maximize win probability.

This is a genuine sequential decision problem (MDP), not a bandit -
actions affect the score differential, which changes the state for the
next possession, exactly like the reward-shaping/state-transition
structure in a full RL formulation.

Team "skill" parameters (three_pt_rate, two_pt_rate, hot_factor) are
derived from real, retrieved team stats (see rl/team_params.py) so the
simulation isn't just abstract numbers - it's calibrated to the actual
team being analyzed. This is documented as a modeling simplification in
the README: the free-tier API doesn't expose shot-location data, so shot
success rates are estimated from scoring efficiency rather than measured
directly.

Actions:
  0 - THREE_POINTER: higher variance, higher reward per make
  1 - TWO_POINTER: lower variance, standard efficiency
  2 - FOUL_STRATEGY: intentionally foul to get the ball back (only
      sensible when trailing, wastes a possession if used incorrectly)
  3 - STALL: run the clock, take minimal risk (only sensible when leading)
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Tuple

THREE_POINTER, TWO_POINTER, FOUL_STRATEGY, STALL = 0, 1, 2, 3
ACTIONS = [THREE_POINTER, TWO_POINTER, FOUL_STRATEGY, STALL]
ACTION_NAMES = {0: "three_pointer", 1: "two_pointer", 2: "foul_strategy", 3: "stall"}


@dataclass
class TeamSkill:
    """Skill parameters derived from real team stats. See rl/team_params.py."""
    three_pt_success: float  # probability a 3PT attempt succeeds
    two_pt_success: float    # probability a 2PT attempt succeeds
    hot_factor: float        # multiplier from recent form (>1 = hot, <1 = cold)
    opp_three_pt_allowed: float
    opp_two_pt_allowed: float


def score_diff_to_bucket(diff: int) -> int:
    """Bucket score differential into a small discrete state space."""
    if diff <= -9:
        return 0   # large deficit
    if diff <= -1:
        return 1   # close deficit
    if diff == 0:
        return 2   # tied
    if diff <= 8:
        return 3   # close lead
    return 4       # large lead


NUM_SCORE_BUCKETS = 5


class ClutchPossessionEnv:
    """
    One episode = a fixed number of remaining possessions in a close game.
    State = (score_diff_bucket, possessions_remaining).
    """

    def __init__(self, team_skill: TeamSkill, opponent_skill: TeamSkill,
                 num_possessions: int = 6, starting_diff_range: Tuple[int, int] = (-6, 6)):
        self.team_skill = team_skill
        self.opponent_skill = opponent_skill
        self.num_possessions = num_possessions
        self.starting_diff_range = starting_diff_range
        self.reset()

    def reset(self) -> Tuple[int, int]:
        self.score_diff = random.randint(*self.starting_diff_range)
        self.possessions_left = self.num_possessions
        self._steps_taken = 0
        return self._state()

    def _state(self) -> Tuple[int, int]:
        return (score_diff_to_bucket(self.score_diff), self.possessions_left)

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Returns (next_state, reward, done). Reward is 0 until the final
        possession, then +1/-1/0 for win/loss/tie - classic sparse terminal
        reward, same structure as the win/loss signal in a full game sim.

        Each action has a genuinely different effect on possession economy,
        not just scoring variance, so the optimal choice actually depends
        on the state (score differential x possessions remaining):
          - THREE/TWO: normal shot, then opponent gets their usual counter
            possession (consumes 1 possession for each side).
          - FOUL_STRATEGY: team doesn't attempt a shot, but the possession
            isn't consumed either - it manufactures an extra possession
            later, at the cost of the opponent getting near-certain free
            throws now. Only worth it when trailing and running out of
            possessions to otherwise catch up.
          - STALL: team doesn't shoot, but crucially the opponent does NOT
            get their counter-possession either (clock control) - a pure
            denial move, only worth it when protecting a lead.
        """
        s = self.team_skill
        opp = self.opponent_skill
        hot = s.hot_factor

        opponent_gets_counter = True

        if action == THREE_POINTER:
            made = random.random() < min(0.95, s.three_pt_success * hot)
            self.score_diff += 3 if made else 0
        elif action == TWO_POINTER:
            made = random.random() < min(0.95, s.two_pt_success * hot)
            self.score_diff += 2 if made else 0
        elif action == FOUL_STRATEGY:
            # Sacrifice this possession's scoring to manufacture an extra
            # possession later; opponent very likely scores off free throws.
            # Capped so the agent can't spam this to stall the episode forever.
            if random.random() < 0.75:
                self.score_diff -= 2
            if self.possessions_left < self.num_possessions + 3:
                self.possessions_left += 1  # buys an extra possession, up to a cap
            opponent_gets_counter = False  # already modeled opponent scoring above
        elif action == STALL:
            # Denial: skip own shot AND deny the opponent their counter.
            opponent_gets_counter = False

        if opponent_gets_counter:
            opp_three = random.random() < opp.three_pt_success
            opp_two = random.random() < opp.two_pt_success
            if random.random() < 0.5:
                self.score_diff -= 3 if opp_three else (2 if opp_two else 0)

        self.possessions_left -= 1
        self._steps_taken += 1
        done = self.possessions_left <= 0 or self._steps_taken >= 30  # hard safety cap

        reward = 0.0
        if done:
            if self.score_diff > 0:
                reward = 1.0
            elif self.score_diff < 0:
                reward = -1.0
            else:
                reward = 0.0

        return self._state(), reward, done

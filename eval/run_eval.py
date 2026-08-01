"""
End-to-end evaluation runner for RivalIQ.

Runs the full pipeline for a configurable list of team matchups:
  1. Pull real team/game data (cached) from balldontlie
  2. Derive team analytics + build scouting notes (RAG corpus)
  3. Generate a grounded matchup report via Groq (free, default) or Claude
  4. Score the report's faithfulness against retrieved stats
  5. Train the RL clutch-possession agent calibrated to these two teams
  6. Compare the trained agent against random and fixed baselines
  7. Save charts (radar + learning curve) and a JSON results summary

Usage:
    export GROQ_API_KEY=gsk_...
    export BALLDONTLIE_API_KEY=...
    python eval/run_eval.py

    # Or benchmark against Claude instead:
    export ANTHROPIC_API_KEY=sk-ant-...
    python eval/run_eval.py --provider anthropic
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.client import BallDontLieClient  # noqa: E402
from data.analytics import build_team_profile  # noqa: E402
from rag.notes import build_corpus  # noqa: E402
from rag.retriever import ScoutingRetriever  # noqa: E402
from rag.generator import ScoutingReportGenerator  # noqa: E402
from eval.faithfulness import check_faithfulness  # noqa: E402
from rl.team_params import derive_team_skill, derive_opponent_skill  # noqa: E402
from rl.environment import ClutchPossessionEnv, TWO_POINTER, THREE_POINTER  # noqa: E402
from rl.agent import QLearningAgent, train, evaluate_random_baseline, evaluate_fixed_baseline  # noqa: E402
from genai.visuals import render_team_comparison_radar, render_learning_curve  # noqa: E402

HERE = Path(__file__).resolve().parent
CHARTS_DIR = HERE / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

MATCHUPS = [("BOS", "LAL"), ("GSW", "MIL")]  # edit to taste
SEASON = 2024

ENV_KEY_NAMES = {"groq": "GROQ_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


def build_profile_for_abbr(client: BallDontLieClient, teams_by_abbr: dict, abbr: str):
    team = teams_by_abbr[abbr]
    games = client.get_games(team_ids=[team["id"]], seasons=[SEASON], max_pages=3)
    return build_team_profile(team, games)


def run_matchup(client: BallDontLieClient, generator: ScoutingReportGenerator,
                 teams_by_abbr: dict, team_a: str, team_b: str) -> dict:
    print(f"\n=== {team_a} vs {team_b} ===")

    profile_a = build_profile_for_abbr(client, teams_by_abbr, team_a)
    profile_b = build_profile_for_abbr(client, teams_by_abbr, team_b)

    corpus = build_corpus([profile_a, profile_b])
    retriever = ScoutingRetriever(corpus)
    retrieved = retriever.retrieve_for_matchup(team_a, team_b)

    report = generator.generate(retrieved)
    print("Generated report:\n", report)

    faith = check_faithfulness(report, retrieved)
    print(f"Faithfulness score: {faith.faithfulness_score} "
          f"({faith.grounded_claims}/{faith.total_numeric_claims} numeric claims grounded)")
    if faith.ungrounded_values:
        print("  Ungrounded values found:", faith.ungrounded_values)

    radar_path = CHARTS_DIR / f"radar_{team_a}_{team_b}.png"
    render_team_comparison_radar(profile_a.to_stats_dict(), profile_b.to_stats_dict(), str(radar_path))

    skill_a = derive_team_skill(profile_a)
    skill_b = derive_opponent_skill(profile_b)
    env = ClutchPossessionEnv(team_skill=skill_a, opponent_skill=skill_b, num_possessions=6)
    agent = QLearningAgent(epsilon_decay_episodes=4000)
    curve = train(env, agent, num_episodes=8000, eval_every=1000, eval_episodes=500)

    baselines = {
        "Random baseline": evaluate_random_baseline(env, 2000),
        "Always 2PT baseline": evaluate_fixed_baseline(env, 2000, TWO_POINTER),
        "Always 3PT baseline": evaluate_fixed_baseline(env, 2000, THREE_POINTER),
    }
    curve_path = CHARTS_DIR / f"learning_curve_{team_a}_{team_b}.png"
    render_learning_curve(curve, baselines, str(curve_path))

    return {
        "matchup": f"{team_a} vs {team_b}",
        "team_a_stats": profile_a.to_stats_dict(),
        "team_b_stats": profile_b.to_stats_dict(),
        "generated_report": report,
        "faithfulness_score": faith.faithfulness_score,
        "faithfulness_detail": {
            "total_numeric_claims": faith.total_numeric_claims,
            "grounded_claims": faith.grounded_claims,
            "ungrounded_values": faith.ungrounded_values,
        },
        "rl_final_win_rate": curve[-1]["win_rate"],
        "rl_baselines": baselines,
        "rl_learning_curve": curve,
        "radar_chart": str(radar_path),
        "learning_curve_chart": str(curve_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run the RivalIQ full-pipeline eval.")
    parser.add_argument(
        "--provider",
        choices=["groq", "anthropic"],
        default="groq",
        help="LLM provider for report generation (default: groq, free tier).",
    )
    args = parser.parse_args()

    llm_env_key = ENV_KEY_NAMES[args.provider]
    if not os.environ.get(llm_env_key):
        print(f"Error: set {llm_env_key} before running eval with provider='{args.provider}'.")
        sys.exit(1)
    if not os.environ.get("BALLDONTLIE_API_KEY"):
        print("Error: set BALLDONTLIE_API_KEY before running eval.")
        sys.exit(1)

    client = BallDontLieClient()
    generator = ScoutingReportGenerator(provider=args.provider)

    teams = client.get_teams()
    teams_by_abbr = {t["abbreviation"]: t for t in teams}

    results = []
    for team_a, team_b in MATCHUPS:
        results.append(run_matchup(client, generator, teams_by_abbr, team_a, team_b))

    out_path = HERE / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {out_path}")
    print(f"Charts written to {CHARTS_DIR}")


if __name__ == "__main__":
    main()

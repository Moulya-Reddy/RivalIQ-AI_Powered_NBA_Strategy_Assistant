"""
GenAI visual layer: renders a radar/spider chart comparing two teams'
derived profiles, and a learning-curve chart for the RL eval report.
Uses matplotlib directly (no external image-gen API needed) so this
runs fully offline and deterministically.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _normalize(value: float, low: float, high: float) -> float:
    return max(0.0, min(1.0, (value - low) / (high - low))) if high != low else 0.5


def render_team_comparison_radar(team_a_stats: Dict, team_b_stats: Dict, save_path: str) -> str:
    """team_a_stats / team_b_stats: output of TeamProfile.to_stats_dict()."""
    categories = ["Win %", "Off. Output", "Defense", "Avg Margin", "Recent Form"]

    def to_radar_values(s: Dict) -> List[float]:
        return [
            s["win_pct"],
            _normalize(s["avg_points_scored"], 95, 125),
            1 - _normalize(s["avg_points_allowed"], 95, 125),  # lower allowed = better, invert
            _normalize(s["avg_margin"], -15, 15),
            _normalize(s["recent_form_margin_last_5"], -15, 15),
        ]

    values_a = to_radar_values(team_a_stats)
    values_b = to_radar_values(team_b_stats)

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values_a += values_a[:1]
    values_b += values_b[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values_a, linewidth=2, label=team_a_stats["abbreviation"], color="#1f77b4")
    ax.fill(angles, values_a, alpha=0.2, color="#1f77b4")
    ax.plot(angles, values_b, linewidth=2, label=team_b_stats["abbreviation"], color="#d62728")
    ax.fill(angles, values_b, alpha=0.2, color="#d62728")

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_yticks([])
    ax.set_title(f"{team_a_stats['abbreviation']} vs {team_b_stats['abbreviation']}", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    fig.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path


def render_learning_curve(curve: List[Dict], baselines: Dict[str, float], save_path: str) -> str:
    episodes = [p["episode"] for p in curve]
    win_rates = [p["win_rate"] for p in curve]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(episodes, win_rates, marker="o", label="RL agent (greedy policy)", color="#2ca02c")
    for name, value in baselines.items():
        ax.axhline(y=value, linestyle="--", label=name, alpha=0.7)

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Win rate (evaluation)")
    ax.set_title("RL Agent Learning Curve vs Baselines")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path

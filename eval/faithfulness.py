"""
RAG faithfulness eval: checks whether numeric claims in a generated report
actually match the ground-truth numbers in the retrieved scouting notes.

This is a lightweight, deterministic faithfulness check (rather than an
LLM-judge) precisely because for structured stats, exact number matching
is a stronger and cheaper signal than asking another model "does this
seem faithful?" - use the simplest sufficient tool for the job.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List

from rag.notes import ScoutingNote


@dataclass
class FaithfulnessResult:
    total_numeric_claims: int
    grounded_claims: int
    ungrounded_values: List[float]

    @property
    def faithfulness_score(self) -> float:
        if self.total_numeric_claims == 0:
            return 1.0  # no numeric claims made, nothing to falsify
        return round(self.grounded_claims / self.total_numeric_claims, 3)


_NUMBER_RE = re.compile(r"[-+]?\d+\.?\d*")


def _extract_numbers(text: str) -> List[float]:
    return [float(n) for n in _NUMBER_RE.findall(text)]


def _ground_truth_numbers(notes: List[ScoutingNote]) -> set:
    """All numeric values that legitimately appear in the retrieved stats,
    including simple derived variants (rounded, absolute value) so we don't
    penalize reasonable restatement."""
    values = set()
    for note in notes:
        for v in note.stats.values():
            if isinstance(v, (int, float)):
                values.add(round(float(v), 1))
                values.add(round(abs(float(v)), 1))
                values.add(round(float(v)))  # integer-rounded variant
                # win_pct (and similarly scaled fractions) are often written
                # out as a percentage, e.g. 0.667 -> "66.7%"
                if 0 <= v <= 1:
                    values.add(round(float(v) * 100, 1))
                    values.add(round(float(v) * 100))
    return values


def check_faithfulness(generated_text: str, notes: List[ScoutingNote],
                        tolerance: float = 0.15) -> FaithfulnessResult:
    """tolerance: allowed absolute difference to still count a number as
    grounded (guards against trivial rounding-format mismatches)."""
    ground_truth = _ground_truth_numbers(notes)
    claimed_numbers = _extract_numbers(generated_text)

    grounded = 0
    ungrounded = []
    for num in claimed_numbers:
        if any(abs(num - gt) <= tolerance for gt in ground_truth):
            grounded += 1
        else:
            ungrounded.append(num)

    return FaithfulnessResult(
        total_numeric_claims=len(claimed_numbers),
        grounded_claims=grounded,
        ungrounded_values=ungrounded,
    )

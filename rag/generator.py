"""
LLM generation layer: turns retrieved ScoutingNotes into a natural-language
matchup report, with an explicit "ground rules" prompt so the model can
only make claims traceable to the retrieved stats - this is the actual
"RAG" behavior, not just stuffing context and hoping.

Two providers are supported:
  - "groq"      (default): Groq's OpenAI-compatible endpoint, free tier,
                 no credit card required (Llama 3.3 70B by default).
  - "anthropic": the real Claude API - pay-as-you-go, no ongoing free tier.
"""

from __future__ import annotations
import os
from typing import List, Optional

from rag.notes import ScoutingNote

DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "anthropic": "claude-sonnet-4-6",
}
ENV_KEY_NAMES = {
    "groq": "GROQ_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

REPORT_SYSTEM_PROMPT = """You are an NBA scouting analyst. You will be given retrieved \
scouting notes (ground-truth stats) for two teams. Write a concise matchup preview.

Strict rules:
- Every statistical claim you make MUST come directly from the provided notes. \
Do not invent stats, records, or trends not present in the notes.
- If the notes don't contain enough information to answer part of the question, say so \
explicitly rather than guessing.
- Cite the specific numbers you're using inline (e.g. "Boston is 2-0 with a +17.5 avg margin").
- Keep the report to 3-4 sentences per team plus a one-sentence overall lean, \
and do not declare a confident winner - frame it as "what the numbers suggest."
"""


class ScoutingReportGenerator:
    def __init__(self, api_key: Optional[str] = None, provider: str = "groq", model: Optional[str] = None):
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"Unknown provider '{provider}'. Use 'groq' or 'anthropic'.")

        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]

        resolved_key = api_key or os.environ.get(ENV_KEY_NAMES[provider])
        if not resolved_key:
            raise ValueError(
                f"No API key provided for provider '{provider}'. "
                f"Pass api_key= or set {ENV_KEY_NAMES[provider]}."
            )

        if provider == "groq":
            # Groq exposes an OpenAI-compatible /v1 endpoint, so the official
            # `openai` SDK works unchanged - just point base_url at Groq.
            from openai import OpenAI

            self.client = OpenAI(api_key=resolved_key, base_url="https://api.groq.com/openai/v1")
        else:
            import anthropic

            self.client = anthropic.Anthropic(api_key=resolved_key)

    def generate(self, notes: List[ScoutingNote], user_question: Optional[str] = None) -> str:
        if not notes:
            return "No scouting notes were retrieved for these teams - check the team names/abbreviations."

        context = "\n\n".join(f"[{n.team_abbreviation}] {n.text}" for n in notes)
        question = user_question or "Write a matchup preview comparing these teams."
        user_content = f"Retrieved scouting notes:\n{context}\n\nTask: {question}"

        if self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=600,
                messages=[
                    {"role": "system", "content": REPORT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            return response.choices[0].message.content or ""
        else:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=600,
                system=REPORT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            return "".join(block.text for block in message.content if block.type == "text")

"""
Retrieval layer for the scouting-note corpus.

Two retrieval modes, since this corpus has both structured entities (30
teams) and free-text content:
- exact/fuzzy team lookup: for "Celtics vs Lakers" style queries, retrieve
  by team abbreviation/name directly - no point doing approximate search
  when the entity is named explicitly.
- TF-IDF cosine similarity: for open-ended queries like "which team has
  the best defense right now", rank notes by textual relevance.

TF-IDF (rather than a neural embedding model) is a deliberate choice here:
the corpus is small (30 short structured documents) and highly keyword-
driven (team names, "win", "margin", "hot/cold"), so a lightweight,
fully-offline, instantly-reproducible retriever is a better engineering
fit than pulling in a large embedding model for marginal gain. This
tradeoff is documented in the README.
"""

from __future__ import annotations
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.notes import ScoutingNote


class ScoutingRetriever:
    def __init__(self, notes: List[ScoutingNote]):
        self.notes = notes
        self._by_abbr = {n.team_abbreviation.upper(): n for n in notes}
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = self._vectorizer.fit_transform([n.text for n in notes]) if notes else None

    def get_by_team(self, abbreviation_or_name: str) -> Optional[ScoutingNote]:
        key = abbreviation_or_name.strip().upper()
        if key in self._by_abbr:
            return self._by_abbr[key]
        # fall back to substring match on full team name in the note text
        for note in self.notes:
            if key.lower() in note.text.lower():
                return note
        return None

    def semantic_search(self, query: str, top_k: int = 3) -> List[ScoutingNote]:
        if self._matrix is None:
            return []
        query_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        ranked_idx = scores.argsort()[::-1][:top_k]
        return [self.notes[i] for i in ranked_idx if scores[i] > 0]

    def retrieve_for_matchup(self, team_a: str, team_b: str) -> List[ScoutingNote]:
        results = []
        for t in (team_a, team_b):
            note = self.get_by_team(t)
            if note:
                results.append(note)
        return results

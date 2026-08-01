"""
Thin client for the balldontlie NBA API (free tier: teams, players, games).

Design choices:
- Every response is cached to disk (cache/) so repeated runs during
  development don't burn the free tier's 5 requests/minute limit.
- Pagination is handled transparently - callers get a flat list back.
- Rate limiting is handled with a small sleep between calls, plus retry
  on 429, rather than assuming the caller will never exceed 5/min.
"""

from __future__ import annotations
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

BASE_URL = "https://api.balldontlie.io/v1"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

MIN_SECONDS_BETWEEN_CALLS = 13  # keeps us under 5 req/min with margin


class BallDontLieClient:
    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True):
        self.api_key = api_key or os.environ.get("BALLDONTLIE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "No balldontlie API key found. Set BALLDONTLIE_API_KEY or pass api_key=."
            )
        self.use_cache = use_cache
        self._last_call_time = 0.0

    def _cache_path(self, endpoint: str, params: Dict[str, Any]) -> Path:
        key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        safe_endpoint = endpoint.strip("/").replace("/", "_")
        return CACHE_DIR / f"{safe_endpoint}_{digest}.json"

    def _throttle(self):
        elapsed = time.time() - self._last_call_time
        if elapsed < MIN_SECONDS_BETWEEN_CALLS:
            time.sleep(MIN_SECONDS_BETWEEN_CALLS - elapsed)

    def _get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        cache_path = self._cache_path(endpoint, params)
        if self.use_cache and cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

        self._throttle()
        headers = {"Authorization": self.api_key}
        url = f"{BASE_URL}/{endpoint.lstrip('/')}"

        for attempt in range(3):
            response = requests.get(url, headers=headers, params=params, timeout=15)
            self._last_call_time = time.time()
            if response.status_code == 429:
                wait = 15 * (attempt + 1)
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            if self.use_cache:
                with open(cache_path, "w") as f:
                    json.dump(data, f)
            return data

        raise RuntimeError(f"Rate limited repeatedly on {endpoint}, giving up.")

    def _get_all_pages(self, endpoint: str, params: Dict[str, Any], max_pages: int = 10) -> List[dict]:
        results: List[dict] = []
        cursor = None
        for _ in range(max_pages):
            page_params = dict(params)
            if cursor is not None:
                page_params["cursor"] = cursor
            data = self._get(endpoint, page_params)
            results.extend(data.get("data", []))
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break
        return results

    # -- Public API --------------------------------------------------------

    def get_teams(self) -> List[dict]:
        """All 30 NBA teams. Cheap, cacheable, effectively static."""
        return self._get_all_pages("teams", {})

    def get_games(
        self,
        team_ids: Optional[List[int]] = None,
        seasons: Optional[List[int]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        postseason: Optional[bool] = None,
        max_pages: int = 10,
    ) -> List[dict]:
        params: Dict[str, Any] = {"per_page": 100}
        if team_ids:
            params["team_ids[]"] = team_ids
        if seasons:
            params["seasons[]"] = seasons
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        if postseason is not None:
            params["postseason"] = str(postseason).lower()
        return self._get_all_pages("games", params, max_pages=max_pages)

    def get_players(self, search: Optional[str] = None, team_ids: Optional[List[int]] = None) -> List[dict]:
        params: Dict[str, Any] = {"per_page": 100}
        if search:
            params["search"] = search
        if team_ids:
            params["team_ids[]"] = team_ids
        return self._get_all_pages("players", params, max_pages=3)

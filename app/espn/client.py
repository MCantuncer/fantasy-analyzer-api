"""Read-only client for ESPN's fantasy basketball API.

ESPN's API is undocumented, so every call to it lives in this module. Private
leagues need the user's ``espn_s2`` and ``SWID`` cookies. They are passed in per
call, used for that one request, and never logged or stored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

BASE_URL = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba"

LEAGUE_VIEWS = ["mSettings", "mTeam", "mRoster", "mStandings", "mMatchupScore", "mScoreboard", "mDraftDetail"]


class EspnError(Exception):
    """Base error. Messages never include cookie values."""

    code = "espn_error"


class LeaguePrivate(EspnError):
    code = "private_league"


class LeagueNotFound(EspnError):
    code = "league_not_found"


class EspnUnavailable(EspnError):
    code = "espn_unavailable"


@dataclass(frozen=True)
class EspnCredentials:
    espn_s2: str
    swid: str

    def __repr__(self) -> str:  # keeps values out of tracebacks and debug output
        return "EspnCredentials(<redacted>)"

    def cookies(self) -> dict[str, str]:
        return {"espn_s2": self.espn_s2, "SWID": self.swid}


class EspnClient:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, timeout: float = 20.0):
        self._transport = transport
        self._timeout = timeout

    async def _get(
        self,
        url: str,
        params: list[tuple[str, str]],
        creds: EspnCredentials | None,
        headers: dict[str, str] | None = None,
    ) -> dict:
        # A fresh client per call keeps cookies from leaking between users.
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=self._timeout,
            cookies=creds.cookies() if creds else None,
            follow_redirects=False,
        ) as http:
            try:
                resp = await http.get(url, params=params, headers=headers)
            except httpx.HTTPError as exc:
                raise EspnUnavailable(f"ESPN request failed: {type(exc).__name__}") from None
        if resp.status_code in (401, 403):
            raise LeaguePrivate("This league is private. ESPN cookies are required.")
        if resp.status_code == 404:
            raise LeagueNotFound("League not found for this season.")
        if resp.status_code >= 400:
            raise EspnUnavailable(f"ESPN returned HTTP {resp.status_code}.")
        data = resp.json()
        # Older seasons come back wrapped in a list.
        if isinstance(data, list):
            if not data:
                raise LeagueNotFound("League not found for this season.")
            data = data[0]
        return data

    def _league_url(self, season: int, league_id: int) -> str:
        return f"{BASE_URL}/seasons/{season}/segments/0/leagues/{league_id}"

    async def league(self, season: int, league_id: int, creds: EspnCredentials | None) -> dict:
        params = [("view", v) for v in LEAGUE_VIEWS]
        return await self._get(self._league_url(season, league_id), params, creds)

    async def free_agents(
        self, season: int, league_id: int, creds: EspnCredentials | None, limit: int = 150
    ) -> list[dict]:
        flt = {
            "players": {
                "filterStatus": {"value": ["FREEAGENT", "WAIVERS"]},
                "limit": limit,
                "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
            }
        }
        data = await self._get(
            self._league_url(season, league_id),
            [("view", "kona_player_info")],
            creds,
            headers={"X-Fantasy-Filter": json.dumps(flt)},
        )
        return data.get("players", [])

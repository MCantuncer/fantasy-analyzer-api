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

# Light views for the league feed: names, owners, FAAB and the current scoring period.
FEED_VIEWS = ["mTeam", "mSettings", "mStatus"]
# The league-only overview adds this week's category scores.
OVERVIEW_VIEWS = [*FEED_VIEWS, "mMatchupScore"]

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
        unwrap: bool = True,
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
        if unwrap and isinstance(data, list):
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

    async def feed_basics(self, season: int, league_id: int, creds: EspnCredentials | None) -> dict:
        params = [("view", v) for v in FEED_VIEWS]
        return await self._get(self._league_url(season, league_id), params, creds)

    async def overview(self, season: int, league_id: int, creds: EspnCredentials | None) -> dict:
        params = [("view", v) for v in OVERVIEW_VIEWS]
        return await self._get(self._league_url(season, league_id), params, creds)

    async def transactions(
        self, season: int, league_id: int, scoring_period: int, creds: EspnCredentials | None
    ) -> list[dict]:
        """Every transaction in one scoring period (a day): adds, drops, waiver bids won and lost, trades."""
        params = [("view", "mTransactions2"), ("scoringPeriodId", str(scoring_period))]
        data = await self._get(self._league_url(season, league_id), params, creds)
        return data.get("transactions", [])

    async def players(
        self, season: int, league_id: int, player_ids: list[int], creds: EspnCredentials | None
    ) -> list[dict]:
        # ESPN refuses a "limit" without a sort; filterIds alone returns every id asked for.
        flt = {"players": {"filterIds": {"value": player_ids}}}
        data = await self._get(
            self._league_url(season, league_id),
            [("view", "kona_player_info")],
            creds,
            headers={"X-Fantasy-Filter": json.dumps(flt)},
        )
        return data.get("players", [])

    async def day_stats(
        self, season: int, league_id: int, day: int, creds: EspnCredentials | None, limit: int = 1000
    ) -> list[dict]:
        """Every player's game line for one scoring period (a day). The lines are the same in every league."""
        flt = {
            "players": {
                "limit": limit,
                "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
                "filterStatsForCurrentSeasonScoringPeriodId": {"value": [day]},
            }
        }
        data = await self._get(
            self._league_url(season, league_id),
            [("view", "kona_player_info"), ("scoringPeriodId", str(day))],
            creds,
            headers={"X-Fantasy-Filter": json.dumps(flt)},
        )
        return data.get("players", [])

    async def rosters_on_day(self, season: int, league_id: int, day: int, creds: EspnCredentials | None) -> dict:
        """Every team's roster and lineup slots as they stood on one scoring period, past seasons included."""
        params = [("view", "mRoster"), ("scoringPeriodId", str(day))]
        return await self._get(self._league_url(season, league_id), params, creds)

    async def player_pool(
        self, season: int, league_id: int, creds: EspnCredentials | None, limit: int = 1500
    ) -> list[dict]:
        """The most-owned players with their season stats, rostered or not."""
        flt = {"players": {"limit": limit, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}}
        data = await self._get(
            self._league_url(season, league_id),
            [("view", "kona_player_info")],
            creds,
            headers={"X-Fantasy-Filter": json.dumps(flt)},
        )
        return data.get("players", [])

    async def players_by_id(self, season: int, player_ids: list[int]) -> list[dict]:
        """Name, NBA team and injury status of players across ESPN, by id. No league needed."""
        flt = {"filterIds": {"value": player_ids}}
        data = await self._get(
            f"{BASE_URL}/seasons/{season}/players",
            [("view", "kona_player_news")],
            None,
            headers={"X-Fantasy-Filter": json.dumps(flt)},
            unwrap=False,
        )
        return data if isinstance(data, list) else []

    async def team_gamelogs(self, season: int, pro_team_id: int) -> list[dict]:
        """Every game line this season of the players now on one NBA team (games for other teams included)."""
        flt = {
            "filterProTeamIds": {"value": [pro_team_id]},
            "filterStatsForSplitTypeIds": {"value": [5]},
            "limit": 60,
            "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
        }
        data = await self._get(
            f"{BASE_URL}/seasons/{season}/players",
            [("view", "kona_player_info"), ("scoringPeriodId", "0")],
            None,
            headers={"X-Fantasy-Filter": json.dumps(flt)},
            unwrap=False,
        )
        return data if isinstance(data, list) else []

    async def pro_schedule(self, season: int) -> dict:
        """Every NBA team's games by scoring period. The same for every league, no cookies."""
        return await self._get(f"{BASE_URL}/seasons/{season}", [("view", "proTeamSchedules_wl")], None)

    async def _season_players(self, season: int, view: str, flt: dict) -> list[dict]:
        data = await self._get(
            f"{BASE_URL}/seasons/{season}/players",
            [("view", view)],
            None,
            headers={"X-Fantasy-Filter": json.dumps(flt)},
            unwrap=False,
        )
        return data if isinstance(data, list) else []

    async def player_states(self, season: int, limit: int = 700) -> list[dict]:
        """Injury status and NBA team of the most-owned players across ESPN. Light: no stats."""
        flt = {"limit": limit, "sortPercOwned": {"sortPriority": 1, "sortAsc": False}}
        return await self._season_players(season, "kona_player_news", flt)

    async def trending(self, season: int, limit: int = 15) -> list[dict]:
        """The players added in the most ESPN leagues today, with their ownership change."""
        flt = {
            "filterActive": {"value": True},
            "limit": limit,
            "sortPercChanged": {"sortPriority": 1, "sortAsc": False},
        }
        return await self._season_players(season, "kona_player_info", flt)

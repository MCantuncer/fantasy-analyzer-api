"""HTTP API.

Private-league cookies arrive in the ``X-ESPN-S2`` and ``X-ESPN-SWID`` request
headers, are used for that request's ESPN calls, and are never logged or stored.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import schemas
from app.analysis.report import Analyzer, league_summary
from app.cache import TTLCache, credentials_key
from app.espn.client import EspnClient, EspnCredentials, EspnError, LeagueNotFound, LeaguePrivate
from app.espn.parse import League, parse_league

# httpx logs request lines at INFO; keep its logs quiet so nothing about a request leaks.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

LEAGUE_TTL = float(os.getenv("LEAGUE_CACHE_SECONDS", "600"))
SCHEDULE_TTL = 6 * 3600.0  # the NBA calendar changes rarely (postponements, NBA Cup knockouts)
STATUS_FOR_ERROR = {LeaguePrivate: 401, LeagueNotFound: 404}


def current_season(now: datetime | None = None) -> int:
    """ESPN names a season by the year it ends: 2026-27 is 2027."""
    now = now or datetime.now(UTC)
    return now.year + 1 if now.month >= 8 else now.year


def read_credentials(espn_s2: str | None, swid: str | None) -> EspnCredentials | None:
    espn_s2, swid = (espn_s2 or "").strip(), (swid or "").strip()
    if not espn_s2 and not swid:
        return None
    if not espn_s2 or not swid:
        raise HTTPException(400, detail={"code": "incomplete_cookies", "message": "Send both espn_s2 and SWID."})
    return EspnCredentials(espn_s2=espn_s2, swid=swid)


def create_app(espn: EspnClient | None = None) -> FastAPI:
    app = FastAPI(title="Fantasy Analyzer API", version="0.1.0")
    app.state.espn = espn or EspnClient()
    app.state.leagues = TTLCache(LEAGUE_TTL)
    app.state.reports = TTLCache(LEAGUE_TTL)
    app.state.schedules = TTLCache(SCHEDULE_TTL)

    origins = [o for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-ESPN-S2", "X-ESPN-SWID"],
    )

    @app.exception_handler(EspnError)
    async def espn_error(_: Request, exc: EspnError) -> JSONResponse:
        status = STATUS_FOR_ERROR.get(type(exc), 502)
        return JSONResponse(status_code=status, content={"detail": {"code": exc.code, "message": str(exc)}})

    async def pro_schedule(season: int) -> dict | None:
        """The NBA calendar. Analysis still works without it, on an even number of games a week."""
        cached = app.state.schedules.get(season)
        if cached is None:
            try:
                cached = await app.state.espn.pro_schedule(season)
            except EspnError:
                return None
            app.state.schedules.set(season, cached)
        return cached

    async def load_league(league_id: int, season: int, creds: EspnCredentials | None, with_fa: bool) -> League:
        key = (league_id, season, credentials_key(creds), with_fa)
        league = app.state.leagues.get(key)
        if league is None:
            raw = await app.state.espn.league(season, league_id, creds)
            fas = await app.state.espn.free_agents(season, league_id, creds) if with_fa else []
            schedule = await pro_schedule(season) if with_fa else None
            league = parse_league(raw, fas, schedule)
            app.state.leagues.set(key, league)
        return league

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True}

    @app.post("/api/league", response_model=schemas.LeagueSummary, responses={401: {"model": schemas.ErrorBody}})
    async def league(
        body: schemas.LeagueRequest,
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.LeagueSummary:
        creds = read_credentials(x_espn_s2, x_espn_swid)
        lg = await load_league(body.league_id, body.season or current_season(), creds, with_fa=False)
        return league_summary(lg)

    @app.post("/api/analyze", response_model=schemas.AnalysisReport, responses={401: {"model": schemas.ErrorBody}})
    async def analyze(
        body: schemas.AnalyzeRequest,
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.AnalysisReport:
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        key = (body.league_id, season, body.team_id, tuple(sorted(body.keep_player_ids)), credentials_key(creds))
        cached = app.state.reports.get(key)
        if cached is not None:
            return cached
        lg = await load_league(body.league_id, season, creds, with_fa=True)
        if lg.team(body.team_id) is None:
            raise HTTPException(
                404, detail={"code": "team_not_found", "message": "No team with that ID in this league."}
            )
        summary = league_summary(lg)
        if not summary.supported:
            raise HTTPException(422, detail={"code": "unsupported_league", "message": summary.unsupported_reason})
        report = await run_in_threadpool(lambda: Analyzer(lg, body.team_id, body.keep_player_ids).report())
        app.state.reports.set(key, report)
        return report

    return app


app = create_app()

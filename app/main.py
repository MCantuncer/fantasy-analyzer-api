"""HTTP API.

Private-league cookies arrive in the ``X-ESPN-S2`` and ``X-ESPN-SWID`` request
headers, are used for that request's ESPN calls, and are never logged or stored.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import UTC, datetime

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import schemas
from app.analysis.daily import trade_of_the_day
from app.analysis.report import Analyzer, league_summary
from app.cache import TTLCache, credentials_key
from app.espn.client import EspnClient, EspnCredentials, EspnError, LeagueNotFound, LeaguePrivate
from app.espn.parse import League, parse_league
from app.store import Store, TradeKey, create_store

# httpx logs request lines at INFO; keep its logs quiet so nothing about a request leaks.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

LEAGUE_TTL = float(os.getenv("LEAGUE_CACHE_SECONDS", "600"))
STATUS_FOR_ERROR = {LeaguePrivate: 401, LeagueNotFound: 404}
# How long a trade-of-the-day request waits for the scan before answering "pending".
DAILY_WAIT = float(os.getenv("DAILY_TRADE_WAIT_SECONDS", "20"))


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


def today() -> str:
    return datetime.now(UTC).date().isoformat()


def create_app(espn: EspnClient | None = None, store: Store | None = None) -> FastAPI:
    app = FastAPI(title="Fantasy Analyzer API", version="0.1.0")
    app.state.espn = espn or EspnClient()
    app.state.leagues = TTLCache(LEAGUE_TTL)
    app.state.reports = TTLCache(LEAGUE_TTL)
    app.state.store = store or create_store()
    app.state.daily = TTLCache(36 * 3600)
    app.state.daily_jobs = {}

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

    async def load_league(league_id: int, season: int, creds: EspnCredentials | None, with_fa: bool) -> League:
        key = (league_id, season, credentials_key(creds), with_fa)
        league = app.state.leagues.get(key)
        if league is None:
            raw = await app.state.espn.league(season, league_id, creds)
            fas = await app.state.espn.free_agents(season, league_id, creds) if with_fa else []
            league = parse_league(raw, fas)
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

    async def daily_pick(key: TradeKey, creds: EspnCredentials | None) -> schemas.TradeOfTheDay:
        """Today's pick from memory or the store, or by scanning the league (once, however many ask)."""
        pick = app.state.daily.get(key)
        if pick is None:
            body = await run_in_threadpool(app.state.store.get_trade, key)
            if body is not None:
                pick = schemas.TradeOfTheDay.model_validate(body)
        if pick is None:
            league_id, season, _, day = key
            lg = await load_league(league_id, season, creds, with_fa=True)
            summary = league_summary(lg)
            if not summary.supported:
                raise HTTPException(422, detail={"code": "unsupported_league", "message": summary.unsupported_reason})
            pick = await run_in_threadpool(trade_of_the_day, lg, datetime.fromisoformat(day).date())
            await run_in_threadpool(app.state.store.save_trade, key, pick.model_dump(mode="json", exclude={"votes"}))
            # Another server may have saved first; the first pick of the day stands.
            body = await run_in_threadpool(app.state.store.get_trade, key)
            if body is not None:
                pick = schemas.TradeOfTheDay.model_validate(body)
        app.state.daily.set(key, pick)
        return pick

    async def with_votes(key: TradeKey, pick: schemas.TradeOfTheDay) -> schemas.TradeOfTheDay:
        tally = await run_in_threadpool(app.state.store.tally, key)
        votes = {tid: schemas.PollTally(yes=y, no=n) for tid, (y, n) in tally.items()}
        return pick.model_copy(update={"votes": votes})

    @app.post(
        "/api/trade-of-the-day", response_model=schemas.TradeOfTheDay, responses={401: {"model": schemas.ErrorBody}}
    )
    async def trade_of_day(
        body: schemas.LeagueRequest,
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.TradeOfTheDay:
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        key: TradeKey = (body.league_id, season, credentials_key(creds), today())
        cached = app.state.daily.get(key)
        if cached is not None:
            return await with_votes(key, cached)
        # The scan takes a while on a small server: run it once in the background and
        # tell callers to come back if it isn't done in time.
        job = app.state.daily_jobs.get(key)
        if job is None:
            job = asyncio.create_task(daily_pick(key, creds))
            app.state.daily_jobs[key] = job
            job.add_done_callback(lambda _: app.state.daily_jobs.pop(key, None))
        try:
            pick = await asyncio.wait_for(asyncio.shield(job), timeout=DAILY_WAIT)
        except TimeoutError:
            return schemas.TradeOfTheDay(status="pending", date=key[3], league_id=body.league_id, season=season)
        return await with_votes(key, pick)

    @app.post(
        "/api/trade-of-the-day/vote",
        response_model=schemas.TradeOfTheDay,
        responses={409: {"model": schemas.ErrorBody}},
    )
    async def vote(
        body: schemas.VoteRequest,
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.TradeOfTheDay:
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        key: TradeKey = (body.league_id, season, credentials_key(creds), body.date)
        # Votes count only for today's pick and one of its two teams.
        pick = None
        if body.date == today():
            pick = app.state.daily.get(key)
            if pick is None:
                saved = await run_in_threadpool(app.state.store.get_trade, key)
                pick = schemas.TradeOfTheDay.model_validate(saved) if saved else None
        if pick is None or body.team_id not in {s.team_id for s in pick.sides}:
            raise HTTPException(
                409, detail={"code": "poll_closed", "message": "This poll is closed. Reload for today's trade."}
            )
        await run_in_threadpool(app.state.store.vote, key, body.team_id, body.voter, body.take)
        return await with_votes(key, pick)

    return app


app = create_app()

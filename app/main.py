"""HTTP API.

Private-league cookies arrive in the ``X-ESPN-S2`` and ``X-ESPN-SWID`` request
headers, are used for that request's ESPN calls, and are never logged or stored.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
from collections import defaultdict, deque
from datetime import UTC, date, datetime, timedelta

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import schemas
from app.accounts import (
    PASSWORD_MIN,
    AccountStore,
    EmailTaken,
    FollowedLeague,
    Seat,
    User,
    check_password,
    create_account_store,
    hash_password,
    normalize_email,
    team_of,
    verify_google,
)
from app.analysis.daily import trade_of_the_day
from app.analysis.report import Analyzer, league_summary
from app.analysis.shop import ShopAnalyzer, ShopError
from app.analysis.target import TargetAnalyzer, TargetError
from app.cache import TTLCache, credentials_key
from app.espn.client import EspnClient, EspnCredentials, EspnError, LeagueNotFound, LeaguePrivate
from app.espn.parse import League, parse_league
from app.feed import PlayerName, Transaction, parse_players, parse_transactions
from app.scores import build_day, needs_stats, parse_schedule, settled, week_fixtures
from app.social import (
    CHALLENGE_SECONDS,
    Challenge,
    Post,
    SocialStore,
    create_social_store,
    hash_token,
    new_code,
    read_challenge,
    sign_challenge,
    valid_reaction,
)
from app.store import Store, TradeKey, create_store

# httpx logs request lines at INFO; keep its logs quiet so nothing about a request leaks.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

LEAGUE_TTL = float(os.getenv("LEAGUE_CACHE_SECONDS", "600"))
SCHEDULE_TTL = 6 * 3600.0  # the NBA calendar changes rarely (postponements, NBA Cup knockouts)
STATUS_FOR_ERROR = {LeaguePrivate: 401, LeagueNotFound: 404}
# How long a trade-of-the-day request waits for the scan before answering "pending".
DAILY_WAIT = float(os.getenv("DAILY_TRADE_WAIT_SECONDS", "20"))
# The feed pages through the season a week (seven scoring periods) at a time.
FEED_PAGE_PERIODS = 7
POSTS_PER_MINUTE = 10
# While a game is on, one ESPN call a minute serves every visitor.
LIVE_SCORES_TTL = 60.0
NOT_SIGNED_IN = {"code": "not_signed_in", "message": "Sign in first."}
NOT_LINKED = {"code": "not_linked", "message": "Verify your team in this league first."}
NOT_YOUR_TEAM = {"code": "not_your_team", "message": "Only this team's verified manager can see its analysis."}


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


def bearer(authorization: str | None) -> str | None:
    scheme, _, token = (authorization or "").partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def _iso_date(d: date | None) -> str | None:
    return d.isoformat() if d else None


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def create_app(
    espn: EspnClient | None = None,
    store: Store | None = None,
    social: SocialStore | None = None,
    accounts: AccountStore | None = None,
    google_transport=None,
) -> FastAPI:
    app = FastAPI(title="Fantasy Analyzer API", version="0.1.0")
    app.state.espn = espn or EspnClient()
    app.state.leagues = TTLCache(LEAGUE_TTL)
    app.state.reports = TTLCache(LEAGUE_TTL)
    app.state.store = store or create_store()
    app.state.daily = TTLCache(36 * 3600)
    app.state.daily_jobs = {}
    app.state.social = social or create_social_store()
    app.state.accounts = accounts or create_account_store()
    app.state.google_client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip() or None
    app.state.login_attempts = defaultdict(deque)
    app.state.basics = TTLCache(60)
    app.state.overviews = TTLCache(120)
    app.state.tx_recent = TTLCache(60)
    app.state.tx_past = TTLCache(6 * 3600, max_items=4096)
    app.state.players = TTLCache(24 * 3600, max_items=20000)
    app.state.post_times = defaultdict(deque)
    app.state.schedules = TTLCache(SCHEDULE_TTL)
    app.state.score_days = TTLCache(LIVE_SCORES_TTL)
    app.state.settled_days = TTLCache(7 * 24 * 3600, max_items=512)
    app.state.score_locks = defaultdict(asyncio.Lock)
    app.state.now = lambda: datetime.now(UTC)

    origins = [o for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if o]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization", "X-ESPN-S2", "X-ESPN-SWID"],
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

    # Accounts ------------------------------------------------------------------

    async def current_user(authorization: str | None, required: bool = True) -> User | None:
        token = bearer(authorization)
        user = await run_in_threadpool(app.state.accounts.session_user, hash_token(token)) if token else None
        if user is None and required:
            raise HTTPException(401, detail=NOT_SIGNED_IN)
        return user

    async def seat_for(user: User | None, league_id: int, season: int) -> Seat | None:
        if user is None:
            return None
        team_id = await run_in_threadpool(team_of, app.state.accounts, user.id, league_id, season)
        return Seat(user.id, league_id, season, team_id) if team_id is not None else None

    async def linked(authorization: str | None, league_id: int, season: int, required: bool = False) -> Seat | None:
        seat = await seat_for(await current_user(authorization, required=required), league_id, season)
        if seat is None and required:
            raise HTTPException(403, detail=NOT_LINKED)
        return seat

    async def me_for(user: User) -> schemas.Me:
        accounts = app.state.accounts
        followed = await run_in_threadpool(accounts.followed, user.id)
        teams = await run_in_threadpool(accounts.managed_teams, user.id)
        notices = await run_in_threadpool(accounts.notices, user.id)
        return schemas.Me(
            id=user.id,
            email=user.email,
            has_google=user.has_google,
            leagues=[
                schemas.MyLeague(
                    league_id=f.league_id, season=f.season, name=f.name, team_id=teams.get((f.league_id, f.season))
                )
                for f in followed
            ],
            notices=[schemas.Notice(id=n.id, text=n.text, at=iso(n.created_at)) for n in notices],
        )

    async def signed_in(user: User) -> schemas.AuthResult:
        token = secrets.token_urlsafe(32)
        await run_in_threadpool(app.state.accounts.save_session, hash_token(token), user.id)
        return schemas.AuthResult(token=token, me=await me_for(user))

    def throttle_login(email: str) -> None:
        recent = app.state.login_attempts[email]
        now = time.monotonic()
        while recent and now - recent[0] > 300:
            recent.popleft()
        if len(recent) >= 10:
            raise HTTPException(429, detail={"code": "slow_down", "message": "Too many tries. Wait a few minutes."})
        recent.append(now)

    @app.get("/api/auth/config", response_model=schemas.AuthConfig)
    async def auth_config() -> schemas.AuthConfig:
        return schemas.AuthConfig(google_client_id=app.state.google_client_id)

    @app.post("/api/auth/signup", response_model=schemas.AuthResult, responses={409: {"model": schemas.ErrorBody}})
    async def signup(body: schemas.SignUpRequest) -> schemas.AuthResult:
        email = normalize_email(body.email)
        if len(body.password) < PASSWORD_MIN:
            raise HTTPException(422, detail={"code": "weak_password", "message": "Use at least 8 characters."})
        try:
            user = await run_in_threadpool(app.state.accounts.create_user, email, hash_password(body.password))
        except EmailTaken:
            raise HTTPException(
                409, detail={"code": "email_taken", "message": "That email already has an account. Sign in instead."}
            ) from None
        return await signed_in(user)

    @app.post(
        "/api/auth/login",
        response_model=schemas.AuthResult,
        responses={401: {"model": schemas.ErrorBody}, 429: {"model": schemas.ErrorBody}},
    )
    async def login(body: schemas.SignUpRequest) -> schemas.AuthResult:
        email = normalize_email(body.email)
        throttle_login(email)
        found = await run_in_threadpool(app.state.accounts.login_lookup, email)
        ok = found is not None and await run_in_threadpool(check_password, body.password, found[1])
        if not ok:
            raise HTTPException(
                401, detail={"code": "bad_credentials", "message": "That email and password don't match."}
            )
        return await signed_in(found[0])

    @app.post(
        "/api/auth/google",
        response_model=schemas.AuthResult,
        responses={401: {"model": schemas.ErrorBody}, 503: {"model": schemas.ErrorBody}},
    )
    async def google(body: schemas.GoogleSignIn) -> schemas.AuthResult:
        client_id = app.state.google_client_id
        if not client_id:
            raise HTTPException(503, detail={"code": "google_off", "message": "Google sign-in isn't turned on yet."})
        who = await verify_google(body.credential, client_id, google_transport)
        if who is None:
            raise HTTPException(
                401, detail={"code": "bad_google_token", "message": "Google sign-in failed. Try again."}
            )
        accounts = app.state.accounts
        user = await run_in_threadpool(accounts.user_by_google, who.sub)
        if user is None and who.email:
            # Same verified email as a password account: that account gains Google sign-in.
            found = await run_in_threadpool(accounts.login_lookup, who.email)
            if found is not None:
                user = found[0]
                await run_in_threadpool(accounts.attach_google, user.id, who.sub)
                user.has_google = True
        if user is None:
            user = await run_in_threadpool(accounts.create_user, who.email, None, who.sub)
        return await signed_in(user)

    @app.post("/api/auth/logout")
    async def logout(authorization: str | None = Header(default=None)) -> dict:
        if token := bearer(authorization):
            await run_in_threadpool(app.state.accounts.drop_session, hash_token(token))
        return {"ok": True}

    @app.post("/api/me", response_model=schemas.Me, responses={401: {"model": schemas.ErrorBody}})
    async def me(authorization: str | None = Header(default=None)) -> schemas.Me:
        return await me_for(await current_user(authorization))

    @app.post("/api/me/notices/clear", response_model=schemas.Me)
    async def clear_notices(authorization: str | None = Header(default=None)) -> schemas.Me:
        user = await current_user(authorization)
        await run_in_threadpool(app.state.accounts.clear_notices, user.id)
        return await me_for(user)

    @app.post(
        "/api/my-leagues/add",
        response_model=schemas.Me,
        responses={401: {"model": schemas.ErrorBody}, 404: {"model": schemas.ErrorBody}},
    )
    async def add_league(
        body: schemas.LeagueRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.Me:
        user = await current_user(authorization)
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        basics = await feed_basics(body.league_id, season, creds)
        name = ((basics.get("settings") or {}).get("name") or f"League {body.league_id}").strip()
        await run_in_threadpool(app.state.accounts.follow, user.id, FollowedLeague(body.league_id, season, name))
        return await me_for(user)

    @app.post("/api/my-leagues/remove", response_model=schemas.Me)
    async def remove_league(
        body: schemas.LeagueRequest, authorization: str | None = Header(default=None)
    ) -> schemas.Me:
        user = await current_user(authorization)
        await run_in_threadpool(app.state.accounts.unfollow, user.id, body.league_id, body.season or current_season())
        return await me_for(user)

    @app.post("/api/league", response_model=schemas.LeagueSummary, responses={401: {"model": schemas.ErrorBody}})
    async def league(
        body: schemas.LeagueRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.LeagueSummary:
        await current_user(authorization)
        creds = read_credentials(x_espn_s2, x_espn_swid)
        lg = await load_league(body.league_id, body.season or current_season(), creds, with_fa=False)
        return league_summary(lg)

    @app.post(
        "/api/analyze",
        response_model=schemas.AnalysisReport,
        responses={401: {"model": schemas.ErrorBody}, 403: {"model": schemas.ErrorBody}},
    )
    async def analyze(
        body: schemas.AnalyzeRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.AnalysisReport:
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        # A team's analysis is for its verified manager only.
        seat = await linked(authorization, body.league_id, season, required=True)
        if seat.team_id != body.team_id:
            raise HTTPException(403, detail=NOT_YOUR_TEAM)
        key = (
            body.league_id,
            season,
            body.team_id,
            tuple(sorted(body.keep_player_ids)),
            tuple(sorted(body.not_for_sale_player_ids)),
            credentials_key(creds),
        )
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
        report = await run_in_threadpool(
            lambda: Analyzer(lg, body.team_id, body.keep_player_ids, body.not_for_sale_player_ids).report()
        )
        app.state.reports.set(key, report)
        return report

    @app.post(
        "/api/shop",
        response_model=schemas.ShopReport,
        responses={
            401: {"model": schemas.ErrorBody},
            403: {"model": schemas.ErrorBody},
            422: {"model": schemas.ErrorBody},
        },
    )
    async def shop(
        body: schemas.ShopRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.ShopReport:
        """Best deals across the league for one or more of your players."""
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        seat = await linked(authorization, body.league_id, season, required=True)
        if seat.team_id != body.team_id:
            raise HTTPException(403, detail=NOT_YOUR_TEAM)
        key = (
            "shop",
            body.league_id,
            season,
            body.team_id,
            tuple(body.offer_player_ids),
            tuple(sorted(body.keep_player_ids)),
            tuple(sorted(body.not_for_sale_player_ids)),
            body.faab_max,
            credentials_key(creds),
        )
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
        try:
            analyzer = ShopAnalyzer(
                lg,
                body.team_id,
                body.offer_player_ids,
                body.keep_player_ids,
                body.faab_max,
                body.not_for_sale_player_ids,
            )
        except ShopError as exc:
            raise HTTPException(422, detail={"code": exc.code, "message": str(exc)}) from exc
        report = await run_in_threadpool(analyzer.report)
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
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.TradeOfTheDay:
        await current_user(authorization)
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
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.TradeOfTheDay:
        await current_user(authorization)
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

    # League chat ---------------------------------------------------------------

    async def feed_basics(league_id: int, season: int, creds: EspnCredentials | None, fresh: bool = False) -> dict:
        key = (league_id, season, credentials_key(creds))
        raw = None if fresh else app.state.basics.get(key)
        if raw is None:
            raw = await app.state.espn.feed_basics(season, league_id, creds)
            app.state.basics.set(key, raw)
        return raw

    async def period_transactions(
        league_id: int, season: int, period: int, current: int, creds: EspnCredentials | None
    ) -> list[dict]:
        # Past days rarely change, so they are kept for hours; today is refreshed every minute.
        cache = app.state.tx_recent if period >= current else app.state.tx_past
        key = (league_id, season, credentials_key(creds), period)
        raw = cache.get(key)
        if raw is None:
            raw = await app.state.espn.transactions(season, league_id, period, creds)
            cache.set(key, raw)
        return raw

    async def player_names(
        league_id: int, season: int, ids: set[int], creds: EspnCredentials | None
    ) -> dict[int, PlayerName]:
        out = {pid: p for pid in ids if (p := app.state.players.get((season, pid))) is not None}
        missing = sorted(ids - out.keys())
        if missing:
            for pid, p in parse_players(await app.state.espn.players(season, league_id, missing, creds)).items():
                app.state.players.set((season, pid), p)
                out[pid] = p
        return out

    def team_rows(basics: dict) -> list[schemas.FeedTeam]:
        return [
            schemas.FeedTeam(
                id=int(t["id"]),
                name=t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}".strip(),
                abbrev=t.get("abbrev", ""),
                faab_spent=(t.get("transactionCounter") or {}).get("acquisitionBudgetSpent"),
            )
            for t in basics.get("teams", [])
        ]

    def linked_team(seat: Seat, teams: list[schemas.FeedTeam]) -> schemas.LinkedTeam:
        name = next((t.name for t in teams if t.id == seat.team_id), f"Team {seat.team_id}")
        return schemas.LinkedTeam(league_id=seat.league_id, season=seat.season, team_id=seat.team_id, team_name=name)

    def tx_item(tx: Transaction, names: dict[int, PlayerName]) -> schemas.FeedItem:
        def player(pid: int) -> schemas.FeedPlayer:
            p = names.get(pid)
            return schemas.FeedPlayer(id=pid, name=p.name if p else f"Player {pid}", position=p.position if p else "")

        return schemas.FeedItem(
            id=f"tx:{tx.id}",
            kind=tx.kind,
            at=iso(datetime.fromtimestamp(tx.date_ms / 1000, UTC)),
            team_id=tx.team_id,
            team_ids=tx.team_ids,
            bid=tx.bid,
            moves=[
                schemas.FeedMove(
                    player=player(m.player_id), kind=m.kind, from_team_id=m.from_team_id, to_team_id=m.to_team_id
                )
                for m in tx.moves
            ],
            losing_bids=[schemas.FeedBid(team_id=b.team_id, bid=b.bid) for b in tx.losing_bids],
        )

    def post_item(p: Post) -> schemas.FeedItem:
        return schemas.FeedItem(id=f"post:{p.id}", kind="post", at=iso(p.created_at), team_id=p.team_id, body=p.body)

    async def with_social(
        items: list[schemas.FeedItem], league_id: int, season: int, session: Seat
    ) -> list[schemas.FeedItem]:
        targets = [i.id for i in items]
        social = app.state.social
        marks = await run_in_threadpool(social.reactions, league_id, season, targets, session.team_id)
        counts = await run_in_threadpool(social.reply_counts, league_id, season, targets)
        for i in items:
            rc = marks.get(i.id)
            if rc is not None:
                i.reactions = schemas.Reactions(counts=rc.counts, mine=sorted(rc.mine))
            i.replies = counts.get(i.id, 0)
        return items

    @app.post(
        "/api/team-link/start", response_model=schemas.TeamLinkChallenge, responses={404: {"model": schemas.ErrorBody}}
    )
    async def team_link_start(
        body: schemas.TeamLinkStart,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.TeamLinkChallenge:
        user = await current_user(authorization)
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        team = next(
            (t for t in team_rows(await feed_basics(body.league_id, season, creds)) if t.id == body.team_id), None
        )
        if team is None:
            raise HTTPException(
                404, detail={"code": "team_not_found", "message": "No team with that ID in this league."}
            )
        expires = int(time.time()) + CHALLENGE_SECONDS
        c = Challenge(body.league_id, season, body.team_id, new_code(team.abbrev), expires, user.id)
        return schemas.TeamLinkChallenge(
            challenge=sign_challenge(c),
            code=c.code,
            expires_at=iso(datetime.fromtimestamp(c.expires, UTC)),
            team_id=team.id,
            team_name=team.name,
            current_abbrev=team.abbrev,
        )

    @app.post(
        "/api/team-link/verify",
        response_model=schemas.LinkedTeam,
        responses={409: {"model": schemas.ErrorBody}, 410: {"model": schemas.ErrorBody}},
    )
    async def team_link_verify(
        body: schemas.TeamLinkVerify,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.LinkedTeam:
        user = await current_user(authorization)
        creds = read_credentials(x_espn_s2, x_espn_swid)
        c = read_challenge(body.challenge)
        if c is None or c.user_id != user.id:
            raise HTTPException(
                410, detail={"code": "challenge_expired", "message": "That code expired. Get a new one."}
            )
        # Always read ESPN fresh here: the abbreviation was just changed.
        teams = team_rows(await feed_basics(c.league_id, c.season, creds, fresh=True))
        team = next((t for t in teams if t.id == c.team_id), None)
        if team is None or team.abbrev.strip().upper() != c.code:
            now = team.abbrev if team else "?"
            raise HTTPException(
                409,
                detail={
                    "code": "abbrev_mismatch",
                    "message": f"ESPN still shows {now}. Save {c.code} as your abbreviation on ESPN, then check again.",
                },
            )
        accounts = app.state.accounts
        previous = await run_in_threadpool(accounts.set_manager, c.league_id, c.season, c.team_id, user.id)
        basics = await feed_basics(c.league_id, c.season, creds)
        league_name = ((basics.get("settings") or {}).get("name") or f"League {c.league_id}").strip()
        await run_in_threadpool(accounts.follow, user.id, FollowedLeague(c.league_id, c.season, league_name))
        if previous is not None:
            await run_in_threadpool(
                accounts.add_notice,
                previous,
                f"Another account verified {team.name} in {league_name}, so you are no longer its manager here. "
                "If you do manage it, verify it again.",
            )
        return linked_team(Seat(user.id, c.league_id, c.season, c.team_id), teams)

    @app.post("/api/feed", response_model=schemas.FeedPage, responses={401: {"model": schemas.ErrorBody}})
    async def feed(
        body: schemas.FeedRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.FeedPage:
        """A week of moves, newest first. For a verified manager: the chat, reactions and reply counts too."""
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        session = await seat_for(await current_user(authorization), body.league_id, season)
        basics = await feed_basics(body.league_id, season, creds)
        status = basics.get("status") or {}
        current = int(status.get("latestScoringPeriod") or basics.get("scoringPeriodId") or 1)
        first = int(status.get("firstScoringPeriod") or 1)
        top = current - body.page * FEED_PAGE_PERIODS
        periods = [p for p in range(max(first, top - FEED_PAGE_PERIODS + 1), top + 1)]
        raw = await asyncio.gather(*(period_transactions(body.league_id, season, p, current, creds) for p in periods))
        txs = parse_transactions([t for day in raw for t in day])
        names = await player_names(body.league_id, season, {pid for tx in txs for pid in tx.player_ids}, creds)
        items = [tx_item(tx, names) for tx in txs]
        teams = team_rows(basics)
        me = None
        if session is not None:
            now = datetime.now(UTC)
            until = None if body.page == 0 else now - timedelta(days=FEED_PAGE_PERIODS * body.page)
            since = now - timedelta(days=FEED_PAGE_PERIODS * (body.page + 1))
            posts = await run_in_threadpool(app.state.social.chat, body.league_id, season, since, until)
            items += [post_item(p) for p in posts]
            items.sort(key=lambda i: i.at, reverse=True)
            items = await with_social(items, body.league_id, season, session)
            me = linked_team(session, teams)
        acq = (basics.get("settings") or {}).get("acquisitionSettings") or {}
        return schemas.FeedPage(
            league_id=body.league_id,
            season=season,
            page=body.page,
            has_more=bool(periods) and periods[0] > first,
            items=items,
            teams=teams,
            faab_budget=acq.get("acquisitionBudget") if acq.get("isUsingAcquisitionBudget") else None,
            me=me,
        )

    @app.post("/api/feed/replies", response_model=schemas.RepliesPage, responses={401: {"model": schemas.ErrorBody}})
    async def replies(
        body: schemas.RepliesRequest, authorization: str | None = Header(default=None)
    ) -> schemas.RepliesPage:
        season = body.season or current_season()
        session = await linked(authorization, body.league_id, season, required=True)
        posts = await run_in_threadpool(app.state.social.replies, body.league_id, season, body.target)
        items = await with_social([post_item(p) for p in posts], body.league_id, season, session)
        return schemas.RepliesPage(target=body.target, items=items)

    @app.post(
        "/api/posts",
        response_model=schemas.FeedItem,
        responses={401: {"model": schemas.ErrorBody}, 429: {"model": schemas.ErrorBody}},
    )
    async def create_post(
        body: schemas.PostRequest, authorization: str | None = Header(default=None)
    ) -> schemas.FeedItem:
        season = body.season or current_season()
        session = await linked(authorization, body.league_id, season, required=True)
        text = body.body.strip()
        if not text:
            raise HTTPException(422, detail={"code": "empty_post", "message": "Write something first."})
        recent = app.state.post_times[(body.league_id, season, session.team_id)]
        now = time.monotonic()
        while recent and now - recent[0] > 60:
            recent.popleft()
        if len(recent) >= POSTS_PER_MINUTE:
            raise HTTPException(429, detail={"code": "slow_down", "message": "Easy there. Try again in a minute."})
        recent.append(now)
        post = await run_in_threadpool(
            app.state.social.add_post, body.league_id, season, session.team_id, body.target, text
        )
        return post_item(post)

    @app.post("/api/posts/delete", responses={401: {"model": schemas.ErrorBody}, 403: {"model": schemas.ErrorBody}})
    async def delete_post(body: schemas.DeletePostRequest, authorization: str | None = Header(default=None)) -> dict:
        season = body.season or current_season()
        session = await linked(authorization, body.league_id, season, required=True)
        post = await run_in_threadpool(app.state.social.post, body.league_id, season, body.post_id)
        if post is None or post.team_id != session.team_id:
            raise HTTPException(403, detail={"code": "not_yours", "message": "You can only delete your own posts."})
        await run_in_threadpool(app.state.social.delete_post, body.league_id, season, body.post_id)
        return {"ok": True}

    @app.post(
        "/api/reactions",
        response_model=schemas.ReactionsOut,
        responses={401: {"model": schemas.ErrorBody}, 422: {"model": schemas.ErrorBody}},
    )
    async def react(
        body: schemas.ReactRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.ReactionsOut:
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        session = await linked(authorization, body.league_id, season, required=True)
        team_ids = [t.id for t in team_rows(await feed_basics(body.league_id, season, creds))]
        if not valid_reaction(body.reaction, team_ids):
            raise HTTPException(422, detail={"code": "bad_reaction", "message": "Unknown reaction."})
        social = app.state.social
        await run_in_threadpool(
            social.react, body.league_id, season, body.target, session.team_id, body.reaction, body.on
        )
        marks = await run_in_threadpool(social.reactions, body.league_id, season, [body.target], session.team_id)
        rc = marks.get(body.target)
        out = schemas.Reactions(counts=rc.counts, mine=sorted(rc.mine)) if rc else schemas.Reactions()
        return schemas.ReactionsOut(target=body.target, reactions=out)

    @app.post(
        "/api/league-overview", response_model=schemas.LeagueOverview, responses={401: {"model": schemas.ErrorBody}}
    )
    async def league_overview(
        body: schemas.LeagueRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.LeagueOverview:
        """League-level only, for anyone signed in: this week's matchups, the MVP candidate and the league's pulse."""
        user = await current_user(authorization)
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        key = (body.league_id, season, credentials_key(creds))
        raw = app.state.overviews.get(key)
        if raw is None:
            raw = await app.state.espn.overview(season, body.league_id, creds)
            app.state.overviews.set(key, raw)
        seat = await seat_for(user, body.league_id, season)
        managers = await run_in_threadpool(app.state.accounts.managers, body.league_id, season)
        status = raw.get("status") or {}
        period = int(status.get("currentMatchupPeriod") or 1)

        def record(t: dict) -> str:
            r = (t.get("record") or {}).get("overall") or {}
            return f"{r.get('wins', 0)}-{r.get('losses', 0)}" + (f"-{r['ties']}" if r.get("ties") else "")

        teams = [
            schemas.OverviewTeam(
                id=int(t["id"]),
                name=t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}".strip(),
                abbrev=t.get("abbrev", ""),
                record=record(t),
                has_manager=int(t["id"]) in managers,
                mine=seat is not None and seat.team_id == int(t["id"]),
            )
            for t in raw.get("teams", [])
        ]

        def side(m: dict) -> schemas.MatchupSide:
            score = m.get("cumulativeScore") or {}
            return schemas.MatchupSide(
                team_id=int(m["teamId"]),
                wins=int(score.get("wins", 0)),
                losses=int(score.get("losses", 0)),
                ties=int(score.get("ties", 0)),
            )

        matchups = [
            schemas.OverviewMatchup(home=side(m["home"]), away=side(m["away"]))
            for m in raw.get("schedule", [])
            if m.get("matchupPeriodId") == period and "home" in m and "away" in m
        ]
        sides = [(m.home, m.away) for m in matchups] + [(m.away, m.home) for m in matchups]
        mvp = None
        if sides:
            records = {t.id: t.record for t in teams}
            best, opp = max(sides, key=lambda p: (p[0].wins - p[0].losses, p[0].wins))
            if best.wins > best.losses:
                mvp = schemas.MvpCandidate(
                    team_id=best.team_id,
                    opponent_id=opp.team_id,
                    wins=best.wins,
                    losses=best.losses,
                    ties=best.ties,
                    season_record=records.get(best.team_id, ""),
                )

        spent = {
            int(t["id"]): (t.get("transactionCounter") or {}).get("acquisitionBudgetSpent") or 0
            for t in raw.get("teams", [])
        }
        leader = max(spent, key=lambda tid: spent[tid]) if spent and max(spent.values()) > 0 else None
        acq = (raw.get("settings") or {}).get("acquisitionSettings") or {}
        current = int(status.get("latestScoringPeriod") or raw.get("scoringPeriodId") or 1)
        first = int(status.get("firstScoringPeriod") or 1)
        periods = range(max(first, current - FEED_PAGE_PERIODS + 1), current + 1)
        days = await asyncio.gather(*(period_transactions(body.league_id, season, p, current, creds) for p in periods))
        claims = sum(1 for tx in parse_transactions([t for d in days for t in d]) if tx.kind != "trade")
        trades = sum((t.get("transactionCounter") or {}).get("trades") or 0 for t in raw.get("teams", [])) // 2
        pulse = schemas.LeaguePulse(
            faab_leader_id=leader,
            faab_leader_spent=spent.get(leader) if leader is not None else None,
            faab_budget=acq.get("acquisitionBudget") if acq.get("isUsingAcquisitionBudget") else None,
            claims_last_7_days=claims,
            trades_this_season=trades,
            close_matchups=sum(1 for m in matchups if abs(m.home.wins - m.away.wins) <= 1),
        )
        return schemas.LeagueOverview(
            league_id=body.league_id,
            season=season,
            name=((raw.get("settings") or {}).get("name") or f"League {body.league_id}").strip(),
            matchup_period=period,
            teams=teams,
            matchups=matchups,
            mvp=mvp,
            pulse=pulse,
            my_team_id=seat.team_id if seat else None,
        )

    @app.post(
        "/api/target",
        response_model=schemas.TargetReport,
        responses={401: {"model": schemas.ErrorBody}, 403: {"model": schemas.ErrorBody}},
    )
    async def target(
        body: schemas.TargetRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.TargetReport:
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        seat = await linked(authorization, body.league_id, season, required=True)
        if seat.team_id != body.team_id:
            raise HTTPException(403, detail=NOT_YOUR_TEAM)
        lg = await load_league(body.league_id, season, creds, with_fa=True)
        if lg.team(body.team_id) is None:
            raise HTTPException(
                404, detail={"code": "team_not_found", "message": "No team with that ID in this league."}
            )
        summary = league_summary(lg)
        if not summary.supported:
            raise HTTPException(422, detail={"code": "unsupported_league", "message": summary.unsupported_reason})

        def run() -> schemas.TargetReport:
            return TargetAnalyzer(
                lg, body.team_id, body.player_id, body.keep_player_ids, body.not_for_sale_player_ids
            ).target_report()

        try:
            return await run_in_threadpool(run)
        except TargetError as exc:
            raise HTTPException(400, detail={"code": exc.code, "message": str(exc)}) from exc

    async def score_day(season: int, day: int, league_id: int, creds: EspnCredentials | None, sched) -> dict:
        """One day's games. Finished days come from the store; a day in progress is cached for a minute."""
        key = (season, day)
        body = app.state.settled_days.get(key) or app.state.score_days.get(key)
        if body is not None:
            return body
        async with app.state.score_locks[key]:
            body = app.state.settled_days.get(key) or app.state.score_days.get(key)
            if body is not None:
                return body
            body = await run_in_threadpool(app.state.store.get_day, season, day)
            if body is not None:
                app.state.settled_days.set(key, body)
                return body
            now = app.state.now()
            players = (
                await app.state.espn.day_stats(season, league_id, day, creds) if needs_stats(sched, day, now) else []
            )
            body = build_day(sched, day, players, now)
            if settled(sched, day, now):
                await run_in_threadpool(app.state.store.save_day, season, day, body)
                app.state.settled_days.set(key, body)
            else:
                app.state.score_days.set(key, body)
            return body

    @app.post("/api/scores", response_model=schemas.ScoreDay, responses={401: {"model": schemas.ErrorBody}})
    async def scores(
        body: schemas.ScoresRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.ScoreDay:
        """A day of NBA games with scores and box scores, each player tagged with his fantasy team here."""
        await current_user(authorization)
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        lg = await load_league(body.league_id, season, creds, with_fa=False)
        raw = await pro_schedule(season)
        if raw is None:
            raise HTTPException(
                502, detail={"code": "espn_unavailable", "message": "ESPN's NBA schedule is unavailable."}
            )
        sched = parse_schedule(raw)
        now = app.state.now()
        day = body.day or sched.latest_started(now)
        if day not in sched.games:
            raise HTTPException(404, detail={"code": "no_games", "message": "No NBA games that day."})
        data = await score_day(season, day, body.league_id, creds, sched)
        owners = {p.id: t.id for t in lg.teams for p in t.roster}
        games = [
            schemas.Game(
                **{
                    **g,
                    "lines": [schemas.GameLine(**ln, owner_team_id=owners.get(ln["player_id"])) for ln in g["lines"]],
                }
            )
            for g in data["games"]
        ]
        days = [
            schemas.ScoreDayRef(day=d, date=_iso_date(sched.date_of(d)), games=len(gs))
            for d, gs in sorted(sched.games.items())
        ]
        team = lg.team(body.team_id) if body.team_id else None
        roster = team.roster if team else []

        def fixture_side(pro_team_id: int) -> schemas.GameSide:
            abbrev, name = sched.teams.get(pro_team_id, ("", ""))
            return schemas.GameSide(pro_team_id=pro_team_id, abbrev=abbrev, name=name)

        week = [
            schemas.FixtureDay(
                day=d,
                date=_iso_date(sched.date_of(d)),
                games=[
                    schemas.Fixture(
                        id=g.id,
                        start=iso(g.start),
                        away=fixture_side(g.away),
                        home=fixture_side(g.home),
                        my_players=[
                            schemas.FixturePlayer(player_id=p.id, name=p.name, injury_status=p.injury_status)
                            for p in roster
                            if p.pro_team_id in (g.away, g.home)
                        ],
                    )
                    for g in gs
                ],
            )
            for d, gs in week_fixtures(sched, now)
        ]
        return schemas.ScoreDay(
            league_id=body.league_id,
            season=season,
            day=day,
            date=data["date"],
            games=games,
            days=days,
            live=any(g.status == "live" for g in games),
            week=week,
            updated_at=iso(now),
        )

    return app


app = create_app()

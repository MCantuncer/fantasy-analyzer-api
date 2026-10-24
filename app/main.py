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
from app.analysis.daily import trade_of_the_day
from app.analysis.report import Analyzer, league_summary
from app.analysis.shop import ShopAnalyzer, ShopError
from app.analysis.target import TargetAnalyzer, TargetError
from app.cache import TTLCache, credentials_key
from app.espn.client import EspnClient, EspnCredentials, EspnError, LeagueNotFound, LeaguePrivate
from app.espn.parse import League, parse_league
from app.feed import PlayerName, Transaction, parse_players, parse_transactions
from app.scores import build_day, needs_stats, parse_schedule, settled
from app.social import (
    CHALLENGE_SECONDS,
    Challenge,
    Post,
    Session,
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
NOT_LINKED = {"code": "not_linked", "message": "Link your ESPN team to join the league chat."}


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
    espn: EspnClient | None = None, store: Store | None = None, social: SocialStore | None = None
) -> FastAPI:
    app = FastAPI(title="Fantasy Analyzer API", version="0.1.0")
    app.state.espn = espn or EspnClient()
    app.state.leagues = TTLCache(LEAGUE_TTL)
    app.state.reports = TTLCache(LEAGUE_TTL)
    app.state.store = store or create_store()
    app.state.daily = TTLCache(36 * 3600)
    app.state.daily_jobs = {}
    app.state.social = social or create_social_store()
    app.state.basics = TTLCache(60)
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
        responses={401: {"model": schemas.ErrorBody}, 422: {"model": schemas.ErrorBody}},
    )
    async def shop(
        body: schemas.ShopRequest,
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.ShopReport:
        """Best deals across the league for one or more of your players."""
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
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

    async def linked(authorization: str | None, league_id: int, season: int, required: bool = False) -> Session | None:
        token = bearer(authorization)
        session = await run_in_threadpool(app.state.social.session, hash_token(token)) if token else None
        if session is not None and (session.league_id, session.season) != (league_id, season):
            session = None
        if session is None and required:
            raise HTTPException(401, detail=NOT_LINKED)
        return session

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

    def linked_team(session: Session, teams: list[schemas.FeedTeam], token: str | None = None) -> schemas.LinkedTeam:
        name = next((t.name for t in teams if t.id == session.team_id), f"Team {session.team_id}")
        return schemas.LinkedTeam(
            league_id=session.league_id, season=session.season, team_id=session.team_id, team_name=name, token=token
        )

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
        items: list[schemas.FeedItem], league_id: int, season: int, session: Session
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
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.TeamLinkChallenge:
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        team = next(
            (t for t in team_rows(await feed_basics(body.league_id, season, creds)) if t.id == body.team_id), None
        )
        if team is None:
            raise HTTPException(
                404, detail={"code": "team_not_found", "message": "No team with that ID in this league."}
            )
        c = Challenge(body.league_id, season, body.team_id, new_code(team.abbrev), int(time.time()) + CHALLENGE_SECONDS)
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
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.LinkedTeam:
        creds = read_credentials(x_espn_s2, x_espn_swid)
        c = read_challenge(body.challenge)
        if c is None:
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
        token = secrets.token_urlsafe(32)
        session = Session(c.league_id, c.season, c.team_id)
        await run_in_threadpool(app.state.social.save_session, hash_token(token), session)
        return linked_team(session, teams, token)

    @app.post("/api/team-link/me", response_model=schemas.LinkedTeam, responses={401: {"model": schemas.ErrorBody}})
    async def team_link_me(
        body: schemas.LeagueRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.LinkedTeam:
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        session = await linked(authorization, body.league_id, season, required=True)
        return linked_team(session, team_rows(await feed_basics(body.league_id, season, creds)))

    @app.post("/api/team-link/unlink")
    async def team_link_unlink(authorization: str | None = Header(default=None)) -> dict:
        if token := bearer(authorization):
            await run_in_threadpool(app.state.social.drop_session, hash_token(token))
        return {"ok": True}

    @app.post("/api/feed", response_model=schemas.FeedPage, responses={401: {"model": schemas.ErrorBody}})
    async def feed(
        body: schemas.FeedRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.FeedPage:
        """A week of moves, newest first. With a linked team: the chat, reactions and reply counts too."""
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        session = await linked(authorization, body.league_id, season)
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

    @app.post("/api/target", response_model=schemas.TargetReport, responses={401: {"model": schemas.ErrorBody}})
    async def target(
        body: schemas.TargetRequest,
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.TargetReport:
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
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
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.ScoreDay:
        """A day of NBA games with scores and box scores, each player tagged with his fantasy team here."""
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
        return schemas.ScoreDay(
            league_id=body.league_id,
            season=season,
            day=day,
            date=data["date"],
            games=games,
            days=days,
            live=any(g.status == "live" for g in games),
            updated_at=iso(now),
        )

    return app


app = create_app()

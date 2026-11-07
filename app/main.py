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

from fastapi import FastAPI, Header, HTTPException, Request, Response
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
from app.analysis import absence, teammates
from app.analysis.checks import REASON_CHECKS, IdeaChecker, StaleIdea, focus_order
from app.analysis.checks import TITLES as CHECK_TITLES
from app.analysis.daily import trade_of_the_day
from app.analysis.durability import HISTORY_SEASONS, games_played, long_absences, one_off_candidates
from app.analysis.durability import attach as attach_durability
from app.analysis.injury import LAST_SEASON_WEIGHT, Season, parse_gamelogs
from app.analysis.injury import analyze as analyze_absence
from app.analysis.model import CategoryModel
from app.analysis.projection import crowded_days, project_matchup
from app.analysis.report import Analyzer, league_summary
from app.analysis.shop import ShopAnalyzer, ShopError
from app.analysis.target import TargetAnalyzer, TargetError
from app.cache import TTLCache, credentials_key
from app.espn.client import EspnClient, EspnCredentials, EspnError, LeagueNotFound, LeaguePrivate
from app.espn.parse import League, parse_calendar, parse_league
from app.feed import DEFAULT_POSITIONS, PlayerName, Transaction, parse_players, parse_transactions
from app.feedback import Feedback, FeedbackStore, create_feedback_store, deal_key, idea_key, model_version
from app.gameclock import apply_clocks, fetch_clocks
from app.news.extract import timeline
from app.news.facts import PlayerState
from app.news.service import KEEP as NEWS_KEEP
from app.news.service import NewsService, parse_iso
from app.news.service import iso as news_iso
from app.news.sources import BY_ID as NEWS_SOURCES
from app.news.sources import SOURCES as NEWS_SOURCE_LIST
from app.news.verify import FACT_SOURCE, describe_fact
from app.playercard import Card as ParsedCard
from app.playercard import parse_card
from app.reader import DEEP_BATCH_MAX, DeepReader, NoteReader, Reading, create_deep_reader, create_reader
from app.recap import VERSION as RECAP_VERSION
from app.recap import Week as RecapWeek
from app.recap import build_recap, is_final
from app.rules import MAX_RULES, Rule, RuleStore, create_rule_store
from app.saved import Saved, changes, fingerprint, read_saved, report_key, saved_row
from app.scores import Schedule, build_day, has_starters, needs_stats, parse_schedule, settled, week_fixtures
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
from app.timing import Stopwatch
from app.week import category_scores, live_score, roster_pro_teams, week_bounds, week_games

# httpx logs request lines at INFO; keep its logs quiet so nothing about a request leaks.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

LEAGUE_TTL = float(os.getenv("LEAGUE_CACHE_SECONDS", "600"))
SCHEDULE_TTL = 6 * 3600.0  # the NBA calendar changes rarely (postponements, NBA Cup knockouts)
PAST_GAMES_TTL = 7 * 24 * 3600.0  # finished seasons don't change
ADP_TTL = 6 * 3600.0  # ESPN's ADP barely moves once the season's drafts are done
DEEP_READ_MIN_SECONDS = 300.0  # the deep-read cron does nothing if it ran in the last five minutes
STATUS_FOR_ERROR = {LeaguePrivate: 401, LeagueNotFound: 404}
# How long a trade-of-the-day request waits for the scan before answering "pending".
DAILY_WAIT = float(os.getenv("DAILY_TRADE_WAIT_SECONDS", "20"))
# How often the day's pick is checked again for injuries, roster moves and better deals.
DAILY_RECHECK = float(os.getenv("DAILY_TRADE_RECHECK_SECONDS", str(3 * 3600)))
# The feed pages through the season a week (seven scoring periods) at a time.
FEED_PAGE_PERIODS = 7
POSTS_PER_MINUTE = 10
# A team's game logs: this season's change after every game night, last season's never.
GAMELOG_TTL = 3 * 3600.0
MIN_RELEVANT_MINUTES = 15.0
MIN_MINUTES_GAIN = 1.0  # minutes per game; a smaller rise is not worth listing
PAST_GAMELOG_TTL = 7 * 24 * 3600.0
GAMELOG_BATCH = 40  # players per game-log request: every line of a season is ~30 KB a player
IMPACT_METHOD = (
    "Each teammate's minutes and per-minute stats in games his team played without him, against games he "
    "played, this season and last (last season counts a quarter, recent games count more). With few such games "
    "the result leans on a prior that shares his minutes by position and room to grow."
)
IMPACT_TRACK_RECORD = (
    "Checked on 202 absences of regulars in 2025-26, tuned on 2024-25: minutes error 3.9 per game, against 4.2 "
    "for 'no change' and 4.7 for 'his backup gets his minutes'. The top pick was one of the two biggest real "
    "minutes gainers 35% of the time (backup rule 23%, chance about 20%). Stats over a few games are noisy, so "
    "the fantasy value ranking is a guide, not a promise."
)
# While a game is on, one ESPN call a minute serves every visitor.
LIVE_SCORES_TTL = 60.0
PLAYER_CARD_TTL = 300.0  # a card a few people open in a row is one ESPN call; box scores land after games
CARD_NEWS_TTL = 60.0  # player cards share one build of the news stories
CARD_GAMES = 10
CARD_NEWS = 5
# How long a finished analysis job stays around for its last poll (the report itself is cached).
ANALYSIS_JOB_KEEP = 120.0
NOT_SIGNED_IN = {"code": "not_signed_in", "message": "Sign in first."}
NOT_LINKED = {"code": "not_linked", "message": "Verify your team in this league first."}
NOT_YOUR_TEAM = {"code": "not_your_team", "message": "Only this team's verified manager can see its analysis."}


class AnalysisProgress:
    """A background analysis: its task, and how far along it is (written from the worker thread)."""

    def __init__(self) -> None:
        self.task: asyncio.Task = None  # type: ignore[assignment]
        self.progress = 0.0
        self.stage = "Loading your league"
        self.preview: schemas.AnalysisReport | None = None  # the report so far, before its trade lists
        self.saved: Saved | None = None  # the last saved report, shown until this run replaces it
        self.changes: list[schemas.ReportChange] = []  # what moved in the league since it was built

    def update(self, stage: str, progress: float) -> None:
        self.stage = stage
        self.progress = max(self.progress, min(progress, 0.99))

    def show(self, preview: schemas.AnalysisReport) -> None:
        self.preview = preview


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
    news_transport=None,
    social: SocialStore | None = None,
    accounts: AccountStore | None = None,
    google_transport=None,
    feedback: FeedbackStore | None = None,
    reader: NoteReader | None = None,
    deep_reader: DeepReader | None = None,
    rules: RuleStore | None = None,
    clock_transport=None,
) -> FastAPI:
    app = FastAPI(title="Fantasy Analyzer API", version="0.1.0")
    app.state.espn = espn or EspnClient()
    app.state.leagues = TTLCache(LEAGUE_TTL)
    # League loads under way, so a second request for the same league waits for the first.
    app.state.league_loads: dict[tuple, asyncio.Future] = {}
    app.state.reports = TTLCache(LEAGUE_TTL)
    app.state.store = store or create_store()
    app.state.daily = TTLCache(36 * 3600)
    app.state.daily_jobs = {}
    app.state.analysis_jobs: dict[tuple, AnalysisProgress] = {}
    app.state.social = social or create_social_store()
    app.state.accounts = accounts or create_account_store()
    app.state.feedback = feedback or create_feedback_store()
    app.state.reader = reader or create_reader()
    app.state.deep_reader = deep_reader or create_deep_reader()
    app.state.rules = rules or create_rule_store()
    app.state.deep_read_lock = asyncio.Lock()
    app.state.deep_read_last = None
    app.state.google_client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip() or None
    app.state.login_attempts = defaultdict(deque)
    app.state.basics = TTLCache(60)
    app.state.overviews = TTLCache(120)
    app.state.tx_recent = TTLCache(60)
    app.state.tx_past = TTLCache(6 * 3600, max_items=4096)
    app.state.players = TTLCache(24 * 3600, max_items=20000)
    app.state.post_times = defaultdict(deque)
    app.state.schedules = TTLCache(SCHEDULE_TTL)
    app.state.market_adp = TTLCache(ADP_TTL)
    app.state.past_games = TTLCache(PAST_GAMES_TTL)
    # (season, player id) -> his long absences that season; finished seasons don't change.
    app.state.past_absences = TTLCache(PAST_GAMES_TTL, max_items=20_000)
    app.state.score_days = TTLCache(LIVE_SCORES_TTL)
    app.state.week_sits = TTLCache(LEAGUE_TTL)
    # Quarter and clock from ESPN's scoreboard, by date; refreshed as often as live scores.
    app.state.clocks = TTLCache(LIVE_SCORES_TTL)
    app.state.clock_transport = clock_transport
    app.state.settled_days = TTLCache(7 * 24 * 3600, max_items=512)
    app.state.score_locks = defaultdict(asyncio.Lock)
    # Finished weeks never change: (league, season, period) -> recap, backed by the store.
    app.state.recaps = TTLCache(7 * 24 * 3600, max_items=64)
    app.state.recap_locks = defaultdict(asyncio.Lock)
    app.state.now = lambda: datetime.now(UTC)
    app.state.gamelogs = TTLCache(GAMELOG_TTL, max_items=64)
    app.state.past_gamelogs = TTLCache(PAST_GAMELOG_TTL, max_items=64)
    # (season, player id) -> teammates.Measured: what his absence does to his teammates' minutes.
    app.state.absence_minutes = TTLCache(GAMELOG_TTL, max_items=2048)
    app.state.absence_jobs: dict[int, asyncio.Task] = {}
    app.state.cards = TTLCache(PLAYER_CARD_TTL, max_items=2048)
    app.state.card_stories = TTLCache(CARD_NEWS_TTL, max_items=4)
    app.state.news = NewsService(app.state.espn, app.state.store, transport=news_transport)

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

    async def market_adp(season: int) -> dict[int, float] | None:
        """ESPN's snake-draft ADP, for trade values in snake leagues. Analysis still works without it."""
        cached = app.state.market_adp.get(season)
        if cached is None:
            try:
                cached = await app.state.espn.market_adp(season)
            except EspnError:
                return None
            app.state.market_adp.set(season, cached)
        return cached

    async def past_games(season: int) -> list[dict[int, int]]:
        """Games played in the seasons before ``season``, most recent first. Empty when ESPN won't say."""
        out = []
        for past in range(season - 1, season - 1 - HISTORY_SEASONS, -1):
            played = app.state.past_games.get(past)
            if played is None:
                try:
                    played = games_played(await app.state.espn.season_totals(past), past)
                except EspnError:
                    return []
                app.state.past_games.set(past, played)
            out.append(played)
        return out

    async def past_absences(league: League, past: list[dict[int, int]]) -> list[dict[int, list[int]]] | None:
        """Long absences in the seasons before this one, for players who missed half a season in one of them.

        Only those players can have a single major injury set aside (durability.py), so only
        their game logs are fetched. None when ESPN won't say: every absence then counts.
        """
        ids = {p.id for t in league.teams for p in t.roster} | {p.id for p in league.free_agents}
        wanted = sorted(one_off_candidates(past, ids))
        out: list[dict[int, list[int]]] = []
        for k in range(len(past)):
            season = league.season - 1 - k
            # Cached as a 1-tuple: (None,) means ESPN had no game lines for him that season.
            cached = {pid: app.state.past_absences.get((season, pid)) for pid in wanted}
            found = {pid: v[0] for pid, v in cached.items() if v is not None}
            missing = [pid for pid, v in cached.items() if v is None]
            for i in range(0, len(missing), GAMELOG_BATCH):
                batch = missing[i : i + GAMELOG_BATCH]
                try:
                    got = long_absences(await app.state.espn.player_gamelogs(season, batch), season)
                except EspnError:
                    return None
                for pid in batch:
                    found[pid] = got.get(pid)
                    app.state.past_absences.set((season, pid), (found[pid],))
            out.append({pid: v for pid, v in found.items() if v is not None})
        return out

    async def attach_absences(league: League) -> None:
        """Expected games still to miss for every player listed out, from his game lines."""
        out = sorted({p.id for t in league.teams for p in t.roster if absence.is_out(p)})
        out += sorted({p.id for p in league.free_agents if absence.is_out(p)} - set(out))
        if not out:
            return
        season = league.season
        try:
            played = absence.played_days(await player_gamelogs(season, out, past=False), season)
            idle = [pid for pid in out if not played.get(pid)]
            last_run = (
                absence.season_end_run(await player_gamelogs(season - 1, idle, past=True), season - 1) if idle else {}
            )
        except EspnError:
            return  # the flat days-out guess stays
        absence.attach(league, {pid: played.get(pid, set()) for pid in out}, last_run, reported_timelines(league, out))

    def attach_teammate_minutes(league: League) -> None:
        """Teammates of players out for a while get the minutes they gain while he's out (teammates.py).

        Measuring an absence takes two team game logs from ESPN and a regression, far too slow for
        a league load on a small server. So a load never waits for it: it uses the absences already
        measured (cached by player, shared by every league) and starts one background job for the
        rest, which the next load of the league picks up.
        """
        season, moved, missing = league.season, [], []
        for x in teammates.absent_regulars(league):
            measured = app.state.absence_minutes.get((season, x.id))
            if measured is None:
                missing.append((x.pro_team_id, x.id))
            else:
                moved += teammates.shifts(league, x, measured)
        teammates.apply(league, moved)
        if missing:
            start_measuring(league.id, season, missing)

    def start_measuring(league_id: int, season: int, wanted: list[tuple[int, int]]) -> None:
        """One measuring job per season at a time; absences it doesn't cover wait for the next load."""
        job = app.state.absence_jobs.get(season)
        if job is not None and not job.done():
            return
        app.state.absence_jobs[season] = asyncio.create_task(measure_absences(league_id, season, wanted))

    async def measure_absences(league_id: int, season: int, wanted: list[tuple[int, int]]) -> None:
        """Measures each absence in the background, a team at a time, keeping only the small result."""
        await analyses_done(league_id, season)  # the manager waiting on a report goes first
        by_team: dict[int, list[int]] = defaultdict(list)
        for team, pid in wanted:
            by_team[team].append(pid)
        for team, ids in by_team.items():
            try:
                await measure_team(season, team, ids)
            except EspnError:
                continue  # these teammates keep the plain blend; a later load tries again
            except Exception:
                logging.getLogger(__name__).exception("measuring absences failed for NBA team %s", team)

    async def measure_team(season: int, team: int, ids: list[int]) -> None:
        """One team's game logs, parsed, measured for each of its absences, then dropped."""
        lines, info = parse_gamelogs(await peek_gamelogs(season, team, past=False), season)
        last, _ = parse_gamelogs(await peek_gamelogs(season - 1, team, past=True), season - 1)
        now = Season(lines)
        seasons = [now, Season(last, LAST_SEASON_WEIGHT)]
        roster = [p for p in info.values() if p.team == team]
        for pid in ids:
            impact = await run_in_threadpool(analyze_absence, seasons, team, pid, roster)
            app.state.absence_minutes.set((season, pid), teammates.measure(impact, now, team, pid))

    def reported_timelines(league: League, ids: list[int]) -> dict[int, absence.Report]:
        """The latest stored news item giving a timeline for each of these players."""
        cal = league.calendar
        if cal is None:
            return {}
        wanted, out = set(ids), {}
        now = app.state.now()
        items = sorted(app.state.store.news_items(news_iso(now - NEWS_KEEP)), key=lambda d: d["published"])
        for d in items:
            if d.get("kind") != "injury":
                continue
            found = timeline(f"{d.get('title', '')}. {d.get('summary', '')}")
            if found is None:
                continue
            day = (parse_iso(d["published"]).date() - cal.opening).days + 1
            for pid in wanted.intersection(d.get("player_ids") or []):
                out[pid] = absence.Report(day, found, d.get("title", ""))
        return out

    async def player_gamelogs(season: int, ids: list[int], past: bool) -> list[dict]:
        cache = app.state.past_gamelogs if past else app.state.gamelogs
        key = ("players", season, tuple(ids))
        raw = cache.get(key)
        if raw is None:
            raw = await app.state.espn.player_gamelogs(season, ids)
            cache.set(key, raw)
        return raw

    async def load_league(
        league_id: int, season: int, creds: EspnCredentials | None, with_fa: bool, with_calendar: bool = False
    ) -> League:
        """The league, parsed. ``with_fa`` adds free agents and everything the analysis needs;
        ``with_calendar`` only the NBA calendar, durability and absences, for projections."""
        full = with_fa or with_calendar
        key = (league_id, season, credentials_key(creds), with_fa, full)
        # The analysis's league (free agents and all) has everything a projection needs:
        # reuse it, or the load already under way, instead of fetching the league again.
        wider = (league_id, season, credentials_key(creds), True, True) if full and not with_fa else None
        for k in (key, wider):
            if k is None:
                continue
            league = app.state.leagues.get(k)
            if league is not None:
                return league
            loading = app.state.league_loads.get(k)
            if loading is not None:
                return await asyncio.shield(loading)
        task = asyncio.ensure_future(fetch_league(league_id, season, creds, with_fa, full))
        app.state.league_loads[key] = task
        task.add_done_callback(
            lambda t: t.cancelled() or t.exception()
        )  # a failure nobody waits for isn't logged as lost
        try:
            league = await asyncio.shield(task)
        finally:
            if app.state.league_loads.get(key) is task:
                del app.state.league_loads[key]
        return league

    async def fetch_league(
        league_id: int, season: int, creds: EspnCredentials | None, with_fa: bool, full: bool
    ) -> League:
        key = (league_id, season, credentials_key(creds), with_fa, full)
        league = app.state.leagues.get(key)
        if league is None:
            raw = await app.state.espn.league(season, league_id, creds)
            fas = await app.state.espn.free_agents(season, league_id, creds) if with_fa else []
            schedule = await pro_schedule(season) if full else None
            snake = ((raw.get("settings") or {}).get("draftSettings") or {}).get("type") != "AUCTION"
            adp = await market_adp(season) if with_fa and snake else None
            league = parse_league(raw, fas, schedule, adp)
            if full:
                past = await past_games(season)
                attach_durability(league, past, await past_absences(league, past) if past else None)
                await attach_absences(league)
                attach_teammate_minutes(league)
            app.state.leagues.set(key, league)
        return league

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True}

    async def peek_gamelogs(season: int, pro_team_id: int, past: bool) -> list[dict]:
        """A team's game logs from the cache if they're there, else from ESPN without caching them:
        background jobs walk many teams, and the cache keeps up to 64 of these big lists."""
        cache = app.state.past_gamelogs if past else app.state.gamelogs
        raw = cache.get((season, pro_team_id))
        return raw if raw is not None else await app.state.espn.team_gamelogs(season, pro_team_id)

    async def team_gamelogs(season: int, pro_team_id: int, past: bool) -> list[dict]:
        cache = app.state.past_gamelogs if past else app.state.gamelogs
        raw = cache.get((season, pro_team_id))
        if raw is None:
            raw = await app.state.espn.team_gamelogs(season, pro_team_id)
            cache.set((season, pro_team_id), raw)
        return raw

    @app.post(
        "/api/injury-impact",
        response_model=schemas.InjuryImpact,
        responses={401: {"model": schemas.ErrorBody}, 404: {"model": schemas.ErrorBody}},
    )
    async def injury_impact(
        body: schemas.InjuryImpactRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.InjuryImpact:
        """Which teammates gain while a player is out, measured from games his team played without him."""
        await current_user(authorization)
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        found = await app.state.espn.players_by_id(season, [body.player_id])
        if not found:
            raise HTTPException(404, detail={"code": "no_player", "message": "No such NBA player."})
        team = int(found[0].get("proTeamId") or 0)
        if team == 0:
            raise HTTPException(404, detail={"code": "no_team", "message": "He isn't on an NBA team."})
        lines, info = parse_gamelogs(await team_gamelogs(season, team, past=False), season)
        last_lines, _ = parse_gamelogs(await team_gamelogs(season - 1, team, past=True), season - 1)
        # Only his current teammates, by ESPN's current team for each player.
        roster = [p for p in info.values() if p.team == team]
        impact = await run_in_threadpool(
            analyze_absence, [Season(lines), Season(last_lines, LAST_SEASON_WEIGHT)], team, body.player_id, roster
        )
        owners: dict[int, int] = {}
        if body.league_id:
            lg = await load_league(body.league_id, season, creds, with_fa=False)
            owners = {p.id: t.id for t in lg.teams for p in t.roster}
        raw = await pro_schedule(season)
        abbrevs = {
            int(t["id"]): str(t.get("abbrev", "")).upper()
            for t in ((raw or {}).get("settings") or {}).get("proTeams", [])
            if t.get("id")
        }
        on_team = {p.id for p in roster}
        # Fantasy-relevant gains only: more minutes (when measured, by more than the error band, since
        # a change inside the band is noise), a real share of minutes, and better per-game value.
        gainers = [
            b
            for b in impact.beneficiaries
            if b.player_id in on_team
            and b.minutes_change >= MIN_MINUTES_GAIN
            and (b.confidence == "estimate" or b.minutes_change > b.error)
            and b.minutes_after >= MIN_RELEVANT_MINUTES
            and b.value > 0
        ][:6]
        return schemas.InjuryImpact(
            player_id=body.player_id,
            name=str(found[0].get("fullName", "")),
            pro_team=abbrevs.get(team, ""),
            minutes=impact.minutes,
            games_missed=impact.games_missed,
            beneficiaries=[
                schemas.Beneficiary(
                    **{k: v for k, v in b.__dict__.items() if k != "gains"},
                    gains=schemas.CategoryGains(**b.gains),
                    owner_team_id=owners.get(b.player_id),
                )
                for b in gainers
            ],
            method=IMPACT_METHOD,
            track_record=IMPACT_TRACK_RECORD,
        )

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

    async def analysis_key(
        body: schemas.AnalyzeRequest, authorization: str | None, creds: EspnCredentials | None
    ) -> tuple:
        """Check the caller may see this team's analysis and return its cache key."""
        season = body.season or current_season()
        # A team's analysis is for its verified manager only.
        seat = await linked(authorization, body.league_id, season, required=True)
        if seat.team_id != body.team_id:
            raise HTTPException(403, detail=NOT_YOUR_TEAM)
        return (
            body.league_id,
            season,
            body.team_id,
            tuple(sorted(body.keep_player_ids)),
            tuple(sorted(body.not_for_sale_player_ids)),
            credentials_key(creds),
        )

    async def build_report(
        body: schemas.AnalyzeRequest, creds: EspnCredentials | None, key: tuple, job: AnalysisProgress
    ) -> schemas.AnalysisReport:
        season = key[1]
        lg = await load_league(body.league_id, season, creds, with_fa=True)
        if lg.team(body.team_id) is None:
            raise HTTPException(
                404, detail={"code": "team_not_found", "message": "No team with that ID in this league."}
            )
        summary = league_summary(lg)
        if not summary.supported:
            raise HTTPException(422, detail={"code": "unsupported_league", "message": summary.unsupported_reason})
        if job.saved is not None and not job.changes:
            job.changes = changes(job.saved.fingerprint, fingerprint(lg), body.team_id)
        job.update("Reading rosters", 0.02)
        built_at = app.state.now()
        report = await run_in_threadpool(
            lambda: Analyzer(
                lg,
                body.team_id,
                body.keep_player_ids,
                body.not_for_sale_player_ids,
                progress=job.update,
                now=app.state.now(),
            ).report(on_preview=job.show)
        )
        app.state.reports.set(key, report)
        try:
            row = saved_row(report, lg, built_at)
            await run_in_threadpool(
                app.state.store.save_report, report_key(key), body.league_id, season, body.team_id, row
            )
        except Exception:  # the report is still served from memory; it just won't open instantly next time
            logging.getLogger(__name__).warning("could not save the report", exc_info=True)
        return report

    async def saved_report(
        body: schemas.AnalyzeRequest, creds: EspnCredentials | None, key: tuple
    ) -> tuple[schemas.AnalysisReport | None, Saved | None, list[schemas.ReportChange]]:
        """The team's saved report if the league still matches it, else the out-of-date one and what changed.

        The check is one light league read (rosters and injury tags, no free agents), cached
        like any league read, so opening the page costs a second or so instead of a full run.
        """
        try:
            saved = read_saved(await run_in_threadpool(app.state.store.get_report, report_key(key)))
        except Exception:
            logging.getLogger(__name__).warning("could not read the saved report", exc_info=True)
            return None, None, []
        if saved is None:
            return None, None, []
        if not saved.fresh(app.state.now()):
            return None, saved, []  # what changed is worked out once the run has the league
        try:
            lg = await load_league(body.league_id, key[1], creds, with_fa=False)
        except EspnError:
            return None, saved, []  # the run reads the league again and reports the error
        now = fingerprint(lg)
        if now == saved.fingerprint:
            return saved.report, None, []
        return None, saved, changes(saved.fingerprint, now, body.team_id)

    def analysis_job(
        body: schemas.AnalyzeRequest,
        creds: EspnCredentials | None,
        key: tuple,
        retry_failed: bool = False,
        saved: Saved | None = None,
        changed: list[schemas.ReportChange] | None = None,
    ) -> AnalysisProgress:
        """One run per team and settings at a time, however many tabs or refreshes ask for it."""
        job = app.state.analysis_jobs.get(key)
        failed = job is not None and job.task.done() and (job.task.cancelled() or job.task.exception() is not None)
        if job is None or (failed and retry_failed):
            job = AnalysisProgress()
            job.saved, job.changes = saved, list(changed or [])
            job.task = asyncio.create_task(build_report(body, creds, key, job))
            app.state.analysis_jobs[key] = job

            def forget() -> None:
                if app.state.analysis_jobs.get(key) is job:
                    del app.state.analysis_jobs[key]

            def done(task: asyncio.Task) -> None:
                if not task.cancelled():
                    task.exception()  # read here so a failure nobody polls for isn't logged as lost
                # Keep a finished job briefly so the next poll can pick up its result or error.
                asyncio.get_running_loop().call_later(ANALYSIS_JOB_KEEP, forget)

            job.task.add_done_callback(done)
        return job

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
        """The whole analysis in one (slow) request. Prefer /api/analyze/job, which never blocks."""
        creds = read_credentials(x_espn_s2, x_espn_swid)
        key = await analysis_key(body, authorization, creds)
        cached = app.state.reports.get(key)
        if cached is not None:
            return cached
        if key not in app.state.analysis_jobs:
            report, _, _ = await saved_report(body, creds, key)
            if report is not None:
                app.state.reports.set(key, report)
                return report
        return await asyncio.shield(analysis_job(body, creds, key, retry_failed=True).task)

    @app.post(
        "/api/analyze/job",
        response_model=schemas.AnalysisJob,
        responses={401: {"model": schemas.ErrorBody}, 403: {"model": schemas.ErrorBody}},
    )
    async def analyze_job(
        body: schemas.AnalyzeRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.AnalysisJob:
        """Start the team's analysis in the background, or say how far along it is.

        Answers at once. Call again with the same body (every couple of seconds) until
        ``status`` is ``done`` (the report is included) or ``failed``.
        """
        creds = read_credentials(x_espn_s2, x_espn_swid)
        key = await analysis_key(body, authorization, creds)
        cached = app.state.reports.get(key)
        if cached is not None:
            return schemas.AnalysisJob(status="done", progress=1.0, stage="Done", report=cached)
        saved, changed = None, []
        if key not in app.state.analysis_jobs:
            report, saved, changed = await saved_report(body, creds, key)
            if report is not None:
                app.state.reports.set(key, report)
                return schemas.AnalysisJob(status="done", progress=1.0, stage="Done", report=report)
        job = analysis_job(body, creds, key, saved=saved, changed=changed)
        if not job.task.done():
            return schemas.AnalysisJob(
                status="running",
                progress=job.progress,
                stage=job.stage,
                preview=job.preview,
                saved=job.saved.report if job.saved else None,
                changes=job.changes,
            )
        app.state.analysis_jobs.pop(key, None)  # a retry after a failure starts afresh
        exc = job.task.exception()
        if exc is None:
            return schemas.AnalysisJob(status="done", progress=1.0, stage="Done", report=job.task.result())
        if isinstance(exc, HTTPException) and isinstance(exc.detail, dict):
            error = schemas.ErrorBody(**exc.detail)
        elif isinstance(exc, EspnError):
            error = schemas.ErrorBody(code=exc.code, message=str(exc))
        else:
            logging.getLogger(__name__).error("analysis failed", exc_info=exc)
            error = schemas.ErrorBody(code="analysis_failed", message="The analysis failed. Try again.")
        return schemas.AnalysisJob(status="failed", progress=job.progress, stage=job.stage, error=error)

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

    async def saved_pick(key: TradeKey) -> schemas.TradeOfTheDay | None:
        pick = app.state.daily.get(key)
        if pick is None:
            body = await run_in_threadpool(app.state.store.get_trade, key)
            if body is not None:
                pick = schemas.TradeOfTheDay.model_validate(body)
                app.state.daily.set(key, pick)
        return pick

    def uneven_pick(pick: schemas.TradeOfTheDay) -> bool:
        """A 2-for-1 or 1-for-2 pick. Only even trades qualify now, so such a pick is out of date."""
        return bool(pick.sides) and len({len(s.sends) for s in pick.sides}) > 1

    def checked_recently(pick: schemas.TradeOfTheDay) -> bool:
        if not pick.checked_at:
            return False
        return (datetime.now(UTC) - datetime.fromisoformat(pick.checked_at)).total_seconds() < DAILY_RECHECK

    async def daily_pick(
        key: TradeKey, creds: EspnCredentials | None, current: schemas.TradeOfTheDay | None
    ) -> schemas.TradeOfTheDay:
        """Scan the league for the day's pick. On a re-check, ``current`` stays unless something beats it."""
        league_id, season, _, day = key
        await analyses_done(league_id, season)
        lg = await load_league(league_id, season, creds, with_fa=True)
        summary = league_summary(lg)
        if not summary.supported:
            raise HTTPException(422, detail={"code": "unsupported_league", "message": summary.unsupported_reason})
        keep = current.pick_id if current else None
        pick = await run_in_threadpool(trade_of_the_day, lg, datetime.fromisoformat(day).date(), keep)
        if current is not None and current.pick_id and pick.pick_id == current.pick_id:
            pick.generated_at = current.generated_at  # same deal with fresh numbers
        await run_in_threadpool(app.state.store.save_trade, key, pick.model_dump(mode="json", exclude={"votes"}))
        app.state.daily.set(key, pick)
        return pick

    async def analyses_done(league_id: int, season: int) -> None:
        """Wait for this league's running analyses before the daily scan.

        Both are long CPU work on a small server. Run together they share the CPU and both
        finish late; one after the other, the manager on the loading screen gets the report
        first, and the scan (which nobody is waiting on yet) follows.
        """
        running = [
            job.task
            for k, job in list(app.state.analysis_jobs.items())
            if k[0] == league_id and k[1] == season and not job.task.done()
        ]
        if running:
            await asyncio.wait(running)

    def daily_job(key: TradeKey, creds: EspnCredentials | None, current: schemas.TradeOfTheDay | None):
        """One scan per league at a time, however many visitors ask."""
        job = app.state.daily_jobs.get(key)
        if job is None:
            job = asyncio.create_task(daily_pick(key, creds, current))
            app.state.daily_jobs[key] = job

            def done(task: asyncio.Task) -> None:
                app.state.daily_jobs.pop(key, None)
                if not task.cancelled():
                    task.exception()  # a failed re-check is retried on the next visit

            job.add_done_callback(done)
        return job

    async def with_votes(key: TradeKey, pick: schemas.TradeOfTheDay) -> schemas.TradeOfTheDay:
        tally = await run_in_threadpool(app.state.store.tally, key, pick.pick_id or "")
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
        current = await saved_pick(key)
        if current is not None and uneven_pick(current):
            current = None  # saved before picks had to be even trades: scan again
        if current is not None:
            # Serve the current pick right away; a stale one is re-checked in the background.
            if not checked_recently(current):
                daily_job(key, creds, current)
            return await with_votes(key, current)
        # The scan takes a while on a small server: run it in the background and tell
        # callers to come back if it isn't done in time.
        job = daily_job(key, creds, None)
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
        pick = await saved_pick(key) if body.date == today() else None
        if pick is None or body.team_id not in {s.team_id for s in pick.sides}:
            raise HTTPException(
                409, detail={"code": "poll_closed", "message": "This poll is closed. Reload for today's trade."}
            )
        if body.pick_id and body.pick_id != pick.pick_id:
            raise HTTPException(
                409,
                detail={"code": "pick_changed", "message": "Today's trade was updated. Reload to vote on the new one."},
            )
        await run_in_threadpool(app.state.store.vote, key, pick.pick_id or "", body.team_id, body.voter, body.take)
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

    async def overview_games(
        league_id: int, season: int, creds: EspnCredentials | None, now: datetime
    ) -> dict[int, schemas.WeekGames]:
        """Each team's games today and this week. Empty if rosters or the NBA calendar can't be had."""
        raw_schedule = await pro_schedule(season)
        if raw_schedule is None:
            return {}
        try:
            lg = await load_league(league_id, season, creds, with_fa=False)
        except EspnError:
            return {}
        try:
            sched = parse_schedule(raw_schedule)
        except (KeyError, TypeError, ValueError):
            logging.getLogger(__name__).warning("NBA calendar has an unexpected shape; overview sent without games")
            return {}
        out = {}
        for team in lg.teams:
            g = week_games(roster_pro_teams(team.roster), sched, now)
            out[team.id] = schemas.WeekGames(today=g.today, left=g.left, total=g.total)
        return out

    @app.post(
        "/api/matchup-projection",
        response_model=schemas.MatchupProjection | None,
        responses={401: {"model": schemas.ErrorBody}, 403: {"model": schemas.ErrorBody}},
    )
    async def matchup_projection(
        body: schemas.LeagueRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.MatchupProjection | None:
        """Your matchup projected to its last day: who takes each category, and who sits on crowded days.

        Each day the best healthy players with a game fill the lineup slots, for both teams.
        Null without the NBA calendar or a matchup this week.
        """
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        seat = await linked(authorization, body.league_id, season, required=True)
        league = await load_league(body.league_id, season, creds, with_fa=False, with_calendar=True)
        proj = await run_in_threadpool(project_matchup, league, seat.team_id)
        if proj is None:
            return None

        def days(rows: list) -> list[schemas.CrowdedDay]:
            return [
                schemas.CrowdedDay(
                    date=d.date,
                    playing=d.playing,
                    slots=d.slots,
                    sit=[schemas.BenchedPlayer(id=pid, name=name) for pid, name in d.sit],
                )
                for d in rows
            ]

        return schemas.MatchupProjection(
            opponent_id=proj.opponent_id,
            wins=proj.wins,
            losses=proj.losses,
            categories=[
                schemas.ProjectedCategory(stat=c.stat, mine=c.mine, theirs=c.theirs, winner=c.winner, close=c.close)
                for c in proj.categories
            ],
            my_crowded=days(proj.my_crowded),
            their_crowded=days(proj.their_crowded),
        )

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

        def overall(t: dict) -> dict:
            return (t.get("record") or {}).get("overall") or {}

        teams = [
            schemas.OverviewTeam(
                id=int(t["id"]),
                name=t.get("name") or f"{t.get('location', '')} {t.get('nickname', '')}".strip(),
                abbrev=t.get("abbrev", ""),
                record=record(t),
                has_manager=int(t["id"]) in managers,
                mine=seat is not None and seat.team_id == int(t["id"]),
                wins=int(overall(t).get("wins") or 0),
                losses=int(overall(t).get("losses") or 0),
                ties=int(overall(t).get("ties") or 0),
                seed=int(t["playoffSeed"]) if t.get("playoffSeed") else None,
                games_back=overall(t).get("gamesBack"),
            )
            for t in raw.get("teams", [])
        ]
        settings = raw.get("settings") or {}
        stat_ids = [
            int(i["statId"]) for i in (settings.get("scoringSettings") or {}).get("scoringItems", []) if "statId" in i
        ]
        now = app.state.now()
        games = await overview_games(body.league_id, season, creds, now)

        def side(m: dict) -> schemas.MatchupSide:
            score = live_score(m)
            team_id = int(m["teamId"])
            return schemas.MatchupSide(
                team_id=team_id,
                wins=int(score.get("wins", 0)),
                losses=int(score.get("losses", 0)),
                ties=int(score.get("ties", 0)),
                categories=[schemas.CategoryScore(**c.__dict__) for c in category_scores(m, stat_ids)],
                games=games.get(team_id),
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
            name=(settings.get("name") or f"League {body.league_id}").strip(),
            matchup_period=period,
            week_start=week_bounds(now)[0].isoformat(),
            week_end=week_bounds(now)[1].isoformat(),
            playoff_teams=(settings.get("scheduleSettings") or {}).get("playoffTeamCount"),
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

    @app.post(
        "/api/trade-check",
        response_model=schemas.TradeCheck,
        responses={401: {"model": schemas.ErrorBody}, 403: {"model": schemas.ErrorBody}},
    )
    async def trade_check(
        body: schemas.TradeCheckRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.TradeCheck:
        """Check a trade idea against today's league: what holds up, what doesn't, and the evidence.

        Every check runs; the ones the quick reasons point at come first. When the manager has
        given feedback on this deal, the verdicts are kept with it.
        """
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
        idea = body.idea
        deal = deal_key(idea.partner_team_id, [p.id for p in idea.give], [p.id for p in idea.get])
        reading = await note_reading(seat, deal)
        if reading is None:
            order = focus_order(body.reasons)
        else:
            # Ticked reasons first, then what the note points at, then what it implies.
            implied = [c for r in reading.reasons for c in REASON_CHECKS.get(r, [])]
            order = focus_order(body.reasons, extra=[*reading.checks, *implied])
        read_as = schemas.NoteRead(kind=reading.kind, summary=reading.summary) if reading else None

        def run() -> schemas.TradeCheck:
            try:
                checker = IdeaChecker(lg, body.team_id, body.idea, body.keep_player_ids)
            except StaleIdea as exc:
                stale = [
                    schemas.CheckVerdict(check=c, title=CHECK_TITLES[c], status="cant_check", evidence=str(exc))
                    for c in order
                ]
                return schemas.TradeCheck(stale=str(exc), verdicts=stale, read_as=read_as)
            return schemas.TradeCheck(
                playoff_pct_before=round(checker.before, 3),
                playoff_pct_after=round(checker.after, 3),
                verdicts=checker.run(order),
                read_as=read_as,
            )

        result = await run_in_threadpool(run)
        if result.stale is None:
            verdicts = [v.model_dump() for v in result.verdicts]
            await run_in_threadpool(
                app.state.feedback.record_checks, seat.user_id, body.league_id, season, deal, verdicts
            )
        return result

    async def note_reading(seat: Seat, deal: str) -> Reading | None:
        """How the manager's note on this deal reads: read once, then kept with the feedback."""
        entries = await run_in_threadpool(app.state.feedback.for_deal, seat.user_id, seat.league_id, seat.season, deal)
        entry = next((f for f in entries if f.text), None)
        if entry is None:
            return None
        if entry.deep_reading is not None:  # Opus's closer read, when an unclear note got one
            return Reading.model_validate(entry.deep_reading)
        if entry.reading is not None:
            return Reading.model_validate(entry.reading)
        if app.state.reader is None:
            return None
        idea = schemas.TradeIdea(**entry.idea)
        reading = await run_in_threadpool(app.state.reader.read, entry.text, idea, entry.rating, entry.reasons)
        if reading is not None:
            await run_in_threadpool(app.state.feedback.record_reading, entry.id, reading.model_dump())
        return reading

    @app.get("/api/feedback/deep-read")
    async def feedback_deep_read() -> dict:
        """Collects finished closer reads of unclear notes and sends new ones, in batches. Called by a cron.

        Safe to call often: each note is sent once, and nothing runs more than every few minutes.
        """
        reader = app.state.deep_reader
        if reader is None:
            return {"ok": True, "enabled": False}
        async with app.state.deep_read_lock:
            now = app.state.now()
            last = app.state.deep_read_last
            if last is not None and (now - last).total_seconds() < DEEP_READ_MIN_SECONDS:
                return {"ok": True, "enabled": True, "skipped": True}
            app.state.deep_read_last = now
            store = app.state.feedback
            collected = 0
            for batch_id in await run_in_threadpool(store.deep_batches):
                results = await run_in_threadpool(reader.collect, batch_id)
                for feedback_id, deep in (results or {}).items():
                    data = deep.model_dump() if deep is not None else None
                    await run_in_threadpool(store.record_deep, feedback_id, batch_id, data)
                    collected += deep is not None
            pending = await run_in_threadpool(store.pending_deep, DEEP_BATCH_MAX)
            notes = [(f.id, f.text, schemas.TradeIdea(**f.idea), f.rating, f.reasons) for f in pending]
            batch_id = await run_in_threadpool(reader.submit, notes) if notes else None
            if batch_id is not None:
                await run_in_threadpool(store.mark_deep_batch, [f.id for f in pending], batch_id)
            return {"ok": True, "enabled": True, "collected": collected, "sent": len(notes) if batch_id else 0}

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
            if body is not None and has_starters(body):
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

    async def with_clocks(data: dict, day_date) -> dict:
        """Adds the quarter and clock to games in progress, and ends games ESPN has called final."""
        if day_date is None or not any(g["status"] == "live" for g in data["games"]):
            return data
        clocks = app.state.clocks.get(day_date)
        if clocks is None:
            clocks = await fetch_clocks(day_date, app.state.clock_transport)
            app.state.clocks.set(day_date, clocks or {})
        return apply_clocks(data, clocks) if clocks else data

    async def week_sits(
        league_id: int, season: int, creds: EspnCredentials | None, team_id: int
    ) -> dict[str, list[int]]:
        """Who sits on each crowded day left this week, by date: the same lineup the matchup projection uses."""
        key = (league_id, season, credentials_key(creds), team_id)
        cached = app.state.week_sits.get(key)
        if cached is not None:
            return cached
        out: dict[str, list[int]] = {}
        try:
            league = await load_league(league_id, season, creds, with_fa=False, with_calendar=True)
            team = league.team(team_id)

            def compute() -> list:
                model = CategoryModel(league)
                period = league.current_period
                if team is None or not model.cal or period not in model.horizon:
                    return []
                return crowded_days(model, team.roster, period)

            for c in await run_in_threadpool(compute):
                out[c.date] = [pid for pid, _ in c.sit]
        except Exception:  # the grid still shows the games without it
            logging.getLogger(__name__).exception("week sits failed for league %s team %s", league_id, team_id)
        app.state.week_sits.set(key, out)
        return out

    async def score_calendar(season: int) -> Schedule:
        raw = await pro_schedule(season)
        if raw is None:
            raise HTTPException(
                502, detail={"code": "espn_unavailable", "message": "ESPN's NBA schedule is unavailable."}
            )
        return parse_schedule(raw)

    def score_day_of(sched: Schedule, day: int | None, now: datetime) -> int:
        day = day or sched.latest_started(now)
        if day not in sched.games:
            raise HTTPException(404, detail={"code": "no_games", "message": "No NBA games that day."})
        return day

    def week_days(sched: Schedule, now: datetime, roster: list, sits: dict[str, list[int]]) -> list[schemas.FixtureDay]:
        def fixture_side(pro_team_id: int) -> schemas.GameSide:
            abbrev, name = sched.teams.get(pro_team_id, ("", ""))
            return schemas.GameSide(pro_team_id=pro_team_id, abbrev=abbrev, name=name)

        return [
            schemas.FixtureDay(
                day=d,
                date=_iso_date(sched.date_of(d)),
                sits=sits.get(_iso_date(sched.date_of(d)) or "", []),
                games=[
                    schemas.Fixture(
                        id=g.id,
                        start=iso(g.start),
                        away=fixture_side(g.away),
                        home=fixture_side(g.home),
                        my_players=[
                            schemas.FixturePlayer(
                                player_id=p.id, name=p.name, injury_status=p.injury_status, pro_team_id=p.pro_team_id
                            )
                            for p in roster
                            if p.pro_team_id in (g.away, g.home)
                        ],
                    )
                    for g in gs
                ],
            )
            for d, gs in week_fixtures(sched, now)
        ]

    def score_page(
        league_id: int,
        season: int,
        sched: Schedule,
        data: dict,
        owners: dict[int, int],
        week: list[schemas.FixtureDay],
        now: datetime,
        complete: bool,
    ) -> schemas.ScoreDay:
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
            league_id=league_id,
            season=season,
            day=data["day"],
            date=data["date"],
            games=games,
            days=days,
            live=any(g.status == "live" for g in games),
            week=week,
            updated_at=iso(now),
            complete=complete,
        )

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
        sched = await score_calendar(season)
        now = app.state.now()
        day = score_day_of(sched, body.day, now)
        data = await score_day(season, day, body.league_id, creds, sched)
        data = await with_clocks(data, sched.date_of(day))
        owners = {p.id: t.id for t in lg.teams for p in t.roster}
        team = lg.team(body.team_id) if body.team_id else None
        sits = await week_sits(body.league_id, season, creds, team.id) if team and body.with_sits else {}
        week = week_days(sched, now, team.roster if team else [], sits)
        return score_page(body.league_id, season, sched, data, owners, week, now, complete=True)

    @app.post("/api/scores/games", response_model=schemas.ScoreDay, responses={401: {"model": schemas.ErrorBody}})
    async def score_games(
        body: schemas.ScoresRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.ScoreDay:
        """The day's games at once: teams, tip-off and status from the NBA calendar, with no stats call.

        Scores and box scores come along only when the day is already in memory (``complete``);
        otherwise ask ``/api/scores`` for them. Who sits on crowded days is ``/api/scores/sits``.
        """
        await current_user(authorization)
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        sched = await score_calendar(season)
        now = app.state.now()
        day = score_day_of(sched, body.day, now)
        key = (season, day)
        data = app.state.settled_days.get(key) or app.state.score_days.get(key)
        complete = data is not None or not needs_stats(sched, day, now)
        if data is None:
            data = build_day(sched, day, [], now)
            if not complete:  # started games: the scores aren't known yet, not zero
                data = {
                    **data,
                    "games": [
                        {
                            **g,
                            "lines": [],
                            "away": {**g["away"], "score": None},
                            "home": {**g["home"], "score": None},
                        }
                        for g in data["games"]
                    ],
                }
        owners: dict[int, int] = {}
        roster: list = []
        if body.team_id or complete:
            lg = await load_league(body.league_id, season, creds, with_fa=False)
            owners = {p.id: t.id for t in lg.teams for p in t.roster}
            team = lg.team(body.team_id) if body.team_id else None
            roster = team.roster if team else []
        week = week_days(sched, now, roster, {})
        return score_page(body.league_id, season, sched, data, owners, week, now, complete=complete)

    @app.post("/api/scores/sits", response_model=schemas.WeekSits, responses={401: {"model": schemas.ErrorBody}})
    async def score_sits(
        body: schemas.SitsRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.WeekSits:
        """Who sits on each crowded day left this week: the slow part of the Scores page, on its own."""
        await current_user(authorization)
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        sched = await score_calendar(season)
        now = app.state.now()
        sits = await week_sits(body.league_id, season, creds, body.team_id)
        return schemas.WeekSits(
            days=[
                schemas.DaySits(
                    day=d, date=_iso_date(sched.date_of(d)), sits=sits.get(_iso_date(sched.date_of(d)) or "", [])
                )
                for d, _ in week_fixtures(sched, now)
            ]
        )

    async def rosters_by_day(
        league_id: int, season: int, days: list[int], creds: EspnCredentials | None
    ) -> dict[int, dict[int, tuple[int, int]]]:
        """Each day's rosters as (team, lineup slot) per player. Two calls at a time; each reply is
        about 1.5 MB of JSON, so it is boiled down at once and dropped."""
        gate = asyncio.Semaphore(2)

        async def one(day: int) -> tuple[int, dict[int, tuple[int, int]]]:
            async with gate:
                raw = await app.state.espn.rosters_on_day(season, league_id, day, creds)
            seats = {
                int(e["playerId"]): (int(t["id"]), int(e.get("lineupSlotId", 0)))
                for t in raw.get("teams", [])
                for e in (t.get("roster") or {}).get("entries", [])
            }
            return day, seats

        return dict(await asyncio.gather(*(one(d) for d in days)))

    async def recap_body(
        league_id: int, season: int, period: int, days: list[int], lg: League, raw: dict, creds
    ) -> dict:
        """The league's recap of one finished week: from memory, the store, or built once and saved."""
        key = (league_id, season, period)
        body = app.state.recaps.get(key)
        if body is not None:
            return body
        async with app.state.recap_locks[key]:
            body = app.state.recaps.get(key) or await run_in_threadpool(
                app.state.store.get_recap, league_id, season, period
            )
            if body is None or body.get("version") != RECAP_VERSION:
                sched = await score_calendar(season)
                day_lines = {
                    d: [ln for g in (await score_day(season, d, league_id, creds, sched))["games"] for ln in g["lines"]]
                    for d in days
                }
                week = RecapWeek(
                    period=period,
                    days=days,
                    day_lines=day_lines,
                    rosters=await rosters_by_day(league_id, season, days, creds),
                    owners_now={p.id: t.id for t in lg.teams for p in t.roster},
                    categories=[c.stat_id for c in lg.categories],
                    matchups=[m for m in raw.get("schedule", []) if int(m.get("matchupPeriodId", 0)) == period],
                )
                body = await run_in_threadpool(build_recap, week)
                await run_in_threadpool(app.state.store.save_recap, league_id, season, period, body)
            app.state.recaps.set(key, body)
            return body

    @app.post(
        "/api/recap",
        response_model=schemas.WeekRecap,
        responses={401: {"model": schemas.ErrorBody}, 404: {"model": schemas.ErrorBody}},
    )
    async def week_recap(
        body: schemas.RecapRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.WeekRecap:
        """A finished week in review: matchup results, players of the week, the free agents who stood
        out and each team's MVP, best add and bench. Built once per league and week, then stored."""
        await current_user(authorization)
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        # Loading the league first checks this caller can see it (private leagues need cookies),
        # because a stored recap is served without asking ESPN again.
        lg = await load_league(body.league_id, season, creds, with_fa=False)
        key = (body.league_id, season, credentials_key(creds))
        raw = app.state.overviews.get(key)
        if raw is None:
            raw = await app.state.espn.overview(season, body.league_id, creds)
            app.state.overviews.set(key, raw)
        sched_raw = await pro_schedule(season)
        cal = parse_calendar(sched_raw, raw) if sched_raw else None
        if cal is None:
            raise HTTPException(
                502, detail={"code": "espn_unavailable", "message": "ESPN's NBA schedule is unavailable."}
            )
        by_period: dict[int, list[dict]] = defaultdict(list)
        for m in raw.get("schedule", []):
            by_period[int(m.get("matchupPeriodId", 0))].append(m)
        finished = [p for p in sorted(cal.period_days) if is_final(by_period.get(p, []))]
        period = body.period or (finished[-1] if finished else None)
        if period is None or period not in finished:
            raise HTTPException(
                404, detail={"code": "week_not_finished", "message": "No finished week to look back on yet."}
            )

        def bounds(p: int) -> tuple[str | None, str | None]:
            days = cal.period_days[p]
            return _iso_date(cal.date_of(days[0])), _iso_date(cal.date_of(days[-1]))

        days = cal.period_days[period]
        recap = await recap_body(body.league_id, season, period, days, lg, raw, creds)
        names = {t.id: t for t in lg.teams}
        start, end = bounds(period)
        return schemas.WeekRecap(
            league_id=body.league_id,
            season=season,
            start=start,
            end=end,
            categories=[c.name for c in lg.categories],
            **{k: v for k, v in recap.items() if k not in ("days", "teams", "version")},
            teams=[
                schemas.RecapTeam(
                    **t,
                    abbrev=names[t["team_id"]].abbrev if t["team_id"] in names else "",
                    name=names[t["team_id"]].name if t["team_id"] in names else "",
                )
                for t in recap["teams"]
            ],
            weeks=[schemas.RecapWeekRef(period=p, start=bounds(p)[0], end=bounds(p)[1]) for p in finished],
        )

    @app.get("/api/news/refresh")
    async def news_refresh() -> dict:
        """Wakes the server and starts a check of every news source if one is due. Returns nothing about news."""
        refreshing = await app.state.news.ensure_fresh(current_season(), app.state.now())
        return {"ok": True, "refreshing": refreshing}

    @app.post("/api/news", response_model=schemas.NewsPage, responses={401: {"model": schemas.ErrorBody}})
    async def news(
        body: schemas.NewsRequest,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.NewsPage:
        """NBA news from many sources, checked against ESPN's player data and each other before it shows."""
        await current_user(authorization)
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        now = app.state.now()
        service: NewsService = app.state.news
        refreshing = await service.ensure_fresh(season, now)
        owners: dict[int, int] = {}
        if body.league_id:
            lg = await load_league(body.league_id, season, creds, with_fa=False)
            owners = {p.id: t.id for t in lg.teams for p in t.roster}
        raw = await pro_schedule(season)
        teams = {
            int(t["id"]): str(t.get("abbrev", "")).upper()
            for t in ((raw or {}).get("settings") or {}).get("proTeams", [])
            if t.get("id")
        }
        stories = await run_in_threadpool(service.stories, teams, now)
        states: dict[int, PlayerState] = await run_in_threadpool(service.states)
        names = {pid: st.name for pid, st in states.items()}

        def player(pid: int) -> schemas.NewsPlayer:
            st = states.get(pid)
            return schemas.NewsPlayer(
                player_id=pid,
                name=st.name if st else "",
                pro_team=teams.get(st.pro_team_id, "") if st else "",
                injury_status=st.status if st else "",
                owner_team_id=owners.get(pid),
            )

        def link(item) -> schemas.NewsSourceLink:
            src = NEWS_SOURCES.get(item.source_id)
            fact = item.source_id == FACT_SOURCE
            page = f"https://www.espn.com/nba/player/_/id/{item.player_ids[0]}" if item.player_ids else ""
            return schemas.NewsSourceLink(
                source_id=item.source_id,
                name=src.name if src else item.source_id,
                kind=src.kind if src else "tip",
                title=describe_fact(item, names, teams) if fact else item.title,
                url=page if fact else item.url,
                published_at=news_iso(item.published),
                summary=item.summary,
            )

        out = [
            schemas.NewsStory(
                id=s.id,
                kind=s.kind,
                status=s.status,
                headline=s.headline,
                summary=s.summary,
                published_at=news_iso(s.published),
                updated_at=news_iso(s.updated),
                players=[player(pid) for pid in s.player_ids if pid in states],
                # One link per source: its latest item on the story.
                sources=[link(i) for i in {i.source_id: i for i in s.items}.values()],
                severity=s.severity,
                conflict=s.conflict,
            )
            for s in stories
            if body.include_held or s.status != "held"
        ]
        health = await run_in_threadpool(service.health)
        sources = [
            schemas.NewsSourceHealth(source_id=src.id, name=src.name, kind=src.kind, **health.get(src.id, {}))
            for src in NEWS_SOURCE_LIST
        ]
        trending = []
        for p in await run_in_threadpool(service.trending):
            own = p.get("ownership") or {}
            pid = int(p["id"])
            trending.append(
                schemas.TrendingPlayer(
                    player_id=pid,
                    name=p.get("fullName", ""),
                    pro_team=teams.get(int(p.get("proTeamId") or 0), ""),
                    injury_status=str(p.get("injuryStatus") or "ACTIVE"),
                    percent_owned=round(float(own.get("percentOwned", 0)), 1),
                    percent_change=round(float(own.get("percentChange", 0)), 1),
                    owner_team_id=owners.get(pid),
                )
            )
        last = await run_in_threadpool(service.last_refresh)
        return schemas.NewsPage(
            league_id=body.league_id,
            stories=out,
            sources=sources,
            trending=trending,
            updated_at=news_iso(last) if last else None,
            refreshing=refreshing,
        )

    card_loads: dict[tuple, asyncio.Future] = {}

    async def card_entry(league_id: int, season: int, player_id: int, creds: EspnCredentials | None) -> dict:
        """ESPN's card for him, cached; the core and the extras asking at once share one call."""
        key = (league_id, season, credentials_key(creds), player_id)
        entry = app.state.cards.get(key)
        if entry is not None:
            return entry
        pending = card_loads.get(key)
        if pending is None:
            pending = asyncio.ensure_future(
                app.state.espn.player_card(season, league_id, player_id, creds, games=CARD_GAMES)
            )
            card_loads[key] = pending
            pending.add_done_callback(lambda _: card_loads.pop(key, None))
        entry = await asyncio.shield(pending)
        if entry is None:
            raise HTTPException(404, detail={"code": "no_player", "message": "No such NBA player."})
        app.state.cards.set(key, entry)
        return entry

    async def load_card(
        league_id: int, season: int, player_id: int, creds: EspnCredentials | None
    ) -> tuple[ParsedCard, dict[int, str]]:
        """His card and the NBA teams' abbreviations by id."""
        # The player and both calendars in parallel: each is its own ESPN call, and in a row
        # they were most of the wait. Last season's calendar is for a player yet to play.
        entry, raw_now, raw_last = await asyncio.gather(
            card_entry(league_id, season, player_id, creds), pro_schedule(season), pro_schedule(season - 1)
        )
        schedules = {}
        if raw_now:
            schedules[season] = parse_schedule(raw_now)
        played = {int(st.get("seasonId") or 0) for st in (entry.get("player") or {}).get("stats") or []}
        if season - 1 in played and raw_last:
            schedules[season - 1] = parse_schedule(raw_last)
        teams = {tid: abbrev for sched in schedules.values() for tid, (abbrev, _) in sched.teams.items()}
        return parse_card(entry, season, schedules), teams

    async def card_news(season: int, player_id: int, teams: dict[int, str]) -> list[schemas.PlayerNews]:
        # Every card reads the same news; build it once a minute, not once a card.
        stories = app.state.card_stories.get((season,))
        if stories is None:
            service: NewsService = app.state.news
            stories = await run_in_threadpool(service.stories, teams, app.state.now())
            app.state.card_stories.set((season,), stories)
        mine = [s for s in stories if player_id in s.player_ids and s.status != "held"]
        mine.sort(key=lambda s: s.updated, reverse=True)

        def row(s) -> schemas.PlayerNews:
            item = s.items[-1]
            src = NEWS_SOURCES.get(item.source_id)
            fact = item.source_id == FACT_SOURCE
            return schemas.PlayerNews(
                id=s.id,
                kind=s.kind,
                status=s.status,
                headline=s.headline,
                published_at=news_iso(s.published),
                updated_at=news_iso(s.updated),
                source="ESPN" if fact else (src.name if src else item.source_id),
                url=f"https://www.espn.com/nba/player/_/id/{player_id}" if fact else item.url,
                severity=s.severity,
            )

        return [row(s) for s in mine[:CARD_NEWS]]

    async def move_names(
        league_id: int, season: int, ids: set[int], creds: EspnCredentials | None
    ) -> dict[int, PlayerName]:
        try:
            return await player_names(league_id, season, ids, creds) if ids else {}
        except EspnError:
            return {}  # the card still shows his own moves

    def card_ms(at: int) -> str:
        return iso(datetime.fromtimestamp(at / 1000, UTC))

    @app.post(
        "/api/player-card/extras",
        response_model=schemas.PlayerCardExtras,
        responses={401: {"model": schemas.ErrorBody}, 404: {"model": schemas.ErrorBody}},
    )
    async def player_card_extras(
        body: schemas.PlayerCardRequest,
        response: Response,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.PlayerCardExtras:
        """The part of a player card that may take longer: his news, and the names of who moved with him."""
        await current_user(authorization)
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        timer = Stopwatch()
        card, teams = await load_card(body.league_id, season, body.player_id, creds)
        timer.mark("espn")
        other_ids = {pid for m in card.moves for pid, _, _ in m.others}
        news, names = await asyncio.gather(
            card_news(season, body.player_id, teams), move_names(body.league_id, season, other_ids, creds)
        )
        timer.mark("rest")
        response.headers["Server-Timing"] = timer.header()
        return schemas.PlayerCardExtras(news=news, names={pid: p.name for pid, p in names.items()})

    @app.post(
        "/api/player-card",
        response_model=schemas.PlayerCard,
        responses={401: {"model": schemas.ErrorBody}, 404: {"model": schemas.ErrorBody}},
    )
    async def player_card(
        body: schemas.PlayerCardRequest,
        response: Response,
        authorization: str | None = Header(default=None),
        x_espn_s2: str | None = Header(default=None),
        x_espn_swid: str | None = Header(default=None),
    ) -> schemas.PlayerCard:
        """One player at a glance: his latest games, health, news and every move of him in this league."""
        await current_user(authorization)
        creds = read_credentials(x_espn_s2, x_espn_swid)
        season = body.season or current_season()
        timer = Stopwatch()
        card, teams = await load_card(body.league_id, season, body.player_id, creds)
        timer.mark("espn")

        # This season's games; last season's only until he plays one.
        games = [g for g in card.games if g.season == season] or [g for g in card.games if g.season == season - 1]
        other_ids = {pid for m in card.moves for pid, _, _ in m.others}
        if body.core:
            # The card shows now; news and any names not yet known come from /extras alongside.
            others = {pid: p for pid in other_ids if (p := app.state.players.get((season, pid))) is not None}
            mine: list = []
        else:
            mine, others = await asyncio.gather(
                card_news(season, body.player_id, teams), move_names(body.league_id, season, other_ids, creds)
            )
        timer.mark("rest")
        response.headers["Server-Timing"] = timer.header()
        logging.getLogger(__name__).info("player card %s: %s", body.player_id, response.headers["Server-Timing"])

        def stats(d: dict[str, float]) -> schemas.GameStats:
            return schemas.GameStats(**{k: round(v, 1) for k, v in d.items()})

        line = card.season_line
        return schemas.PlayerCard(
            player_id=card.id,
            name=card.name,
            position=DEFAULT_POSITIONS.get(card.position_id, ""),
            pro_team=teams.get(card.pro_team_id, ""),
            injury_status=card.injury_status,
            owner_team_id=card.on_team_id or None,
            on_waivers=card.status == "WAIVERS",
            percent_owned=card.percent_owned,
            season_line=schemas.PlayerSeasonLine(
                season=line.season,
                games=line.games,
                per_game=stats(line.per_game),
                fg_pct=round(line.fg_pct, 3) if line.fg_pct is not None else None,
                ft_pct=round(line.ft_pct, 3) if line.ft_pct is not None else None,
            )
            if line
            else None,
            games=[
                schemas.PlayerGame(
                    game_id=g.game_id,
                    season=g.season,
                    day=g.day,
                    date=card_ms(g.start_ms) if g.start_ms else None,
                    opponent=teams.get(g.opponent_id, ""),
                    home=g.home,
                    stats=stats(g.stats),
                )
                for g in games[:CARD_GAMES]
            ],
            news=mine,
            moves=[
                schemas.PlayerMove(
                    id=m.id,
                    kind=m.kind,
                    via=m.via,
                    at=card_ms(m.at_ms),
                    from_team_id=m.from_team_id,
                    to_team_id=m.to_team_id,
                    bid=m.bid,
                    pick=m.pick,
                    others=[
                        schemas.PlayerMoveOther(
                            player_id=pid, name=others[pid].name if pid in others else "", kind=kind, to_team_id=to
                        )
                        for pid, kind, to in m.others
                    ],
                )
                for m in card.moves
                if m.at_ms
            ],
        )

    def feedback_out(f: Feedback) -> schemas.TradeFeedback:
        return schemas.TradeFeedback(
            id=f.id,
            source=f.source,
            rating=f.rating,
            reasons=f.reasons,
            text=f.text,
            idea=schemas.TradeIdea(**f.idea),
            created_at=iso(f.created_at),
        )

    @app.post(
        "/api/trade-feedback",
        response_model=schemas.TradeFeedback,
        responses={401: {"model": schemas.ErrorBody}, 403: {"model": schemas.ErrorBody}},
    )
    async def trade_feedback(
        body: schemas.TradeFeedbackRequest, authorization: str | None = Header(default=None)
    ) -> schemas.TradeFeedback:
        """Save what a manager thought of a trade idea, with the idea as they saw it.

        Giving feedback on the same deal again replaces the earlier entry.
        """
        season = body.season or current_season()
        seat = await linked(authorization, body.league_id, season, required=True)
        if seat.team_id != body.team_id:
            raise HTTPException(403, detail=NOT_YOUR_TEAM)
        idea = body.idea
        entry = Feedback(
            id=0,
            user_id=seat.user_id,
            league_id=body.league_id,
            season=season,
            team_id=body.team_id,
            source=body.source,
            idea_key=idea_key(body.source, idea.partner_team_id, [p.id for p in idea.give], [p.id for p in idea.get]),
            rating=body.rating,
            reasons=sorted(set(body.reasons)),
            text=body.text.strip(),
            idea=idea.model_dump(),
            model_version=model_version(),
            created_at=app.state.now(),
        )
        saved = await run_in_threadpool(app.state.feedback.save, entry)
        return feedback_out(saved)

    @app.post(
        "/api/trade-feedback/mine",
        response_model=schemas.TradeFeedbackList,
        responses={401: {"model": schemas.ErrorBody}, 403: {"model": schemas.ErrorBody}},
    )
    async def my_trade_feedback(
        body: schemas.TradeFeedbackListRequest, authorization: str | None = Header(default=None)
    ) -> schemas.TradeFeedbackList:
        """The feedback this manager gave in the league, newest first."""
        season = body.season or current_season()
        seat = await linked(authorization, body.league_id, season, required=True)
        if seat.team_id != body.team_id:
            raise HTTPException(403, detail=NOT_YOUR_TEAM)
        rows = await run_in_threadpool(app.state.feedback.mine, seat.user_id, body.league_id, season)
        return schemas.TradeFeedbackList(feedback=[feedback_out(f) for f in rows])

    def rule_out(r: Rule) -> schemas.TradeRule:
        return schemas.TradeRule(
            id=r.id, kind=r.kind, value=r.value, label=r.label, why=r.why, created_at=iso(r.created_at)
        )

    async def rule_seat(authorization: str | None, league_id: int, season: int | None, team_id: int) -> Seat:
        seat = await linked(authorization, league_id, season or current_season(), required=True)
        if seat.team_id != team_id:
            raise HTTPException(403, detail=NOT_YOUR_TEAM)
        return seat

    @app.post(
        "/api/trade-rules",
        response_model=schemas.TradeRule,
        responses={
            400: {"model": schemas.ErrorBody},
            401: {"model": schemas.ErrorBody},
            403: {"model": schemas.ErrorBody},
        },
    )
    async def add_trade_rule(
        body: schemas.TradeRuleRequest, authorization: str | None = Header(default=None)
    ) -> schemas.TradeRule:
        """Adds one of this manager's trade rules. Adding a rule they already have returns it unchanged."""
        seat = await rule_seat(authorization, body.league_id, body.season, body.team_id)
        value = body.value.strip()
        bad = None
        if body.kind == "protect_category" and not value:
            bad = "Say which category to protect."
        elif body.kind == "skip_team" and (not value.isdigit() or int(value) == seat.team_id):
            bad = "Say which other team to skip, by its id."
        elif body.kind == "no_injury_prone":
            value = ""
        if bad:
            raise HTTPException(400, detail={"code": "bad_rule", "message": bad})
        existing = await run_in_threadpool(app.state.rules.mine, seat.user_id, seat.league_id, seat.season)
        if len(existing) >= MAX_RULES and not any((r.kind, r.value) == (body.kind, value) for r in existing):
            raise HTTPException(
                400, detail={"code": "too_many_rules", "message": f"You can keep up to {MAX_RULES} rules."}
            )
        label = {
            "protect_category": f"Protect {value}",
            "skip_team": f"Skip {body.team_name.strip() or 'team ' + value}",
            "no_injury_prone": "No injury-prone players in",
        }[body.kind]
        why = body.why.strip()
        rule = Rule(0, seat.user_id, seat.league_id, seat.season, body.kind, value, label, why, app.state.now())
        return rule_out(await run_in_threadpool(app.state.rules.add, rule))

    @app.post(
        "/api/trade-rules/mine",
        response_model=schemas.TradeRuleList,
        responses={401: {"model": schemas.ErrorBody}, 403: {"model": schemas.ErrorBody}},
    )
    async def my_trade_rules(
        body: schemas.TradeRuleListRequest, authorization: str | None = Header(default=None)
    ) -> schemas.TradeRuleList:
        """This manager's trade rules in the league, oldest first."""
        seat = await rule_seat(authorization, body.league_id, body.season, body.team_id)
        rows = await run_in_threadpool(app.state.rules.mine, seat.user_id, seat.league_id, seat.season)
        return schemas.TradeRuleList(rules=[rule_out(r) for r in rows])

    @app.post(
        "/api/trade-rules/delete",
        response_model=schemas.TradeRuleList,
        responses={401: {"model": schemas.ErrorBody}, 403: {"model": schemas.ErrorBody}},
    )
    async def delete_trade_rule(
        body: schemas.TradeRuleDeleteRequest, authorization: str | None = Header(default=None)
    ) -> schemas.TradeRuleList:
        """Removes one of this manager's rules and returns the ones left."""
        seat = await rule_seat(authorization, body.league_id, body.season, body.team_id)
        await run_in_threadpool(app.state.rules.remove, seat.user_id, body.rule_id)
        rows = await run_in_threadpool(app.state.rules.mine, seat.user_id, seat.league_id, seat.season)
        return schemas.TradeRuleList(rules=[rule_out(r) for r in rows])

    return app


app = create_app()

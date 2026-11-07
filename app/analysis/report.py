"""Builds the full analysis for one team: category outlook, builds, trades, pickups, season."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np

from app import schemas
from app.analysis.badges import OwnerPremium, badge_refs
from app.analysis.durability import injury_prone
from app.analysis.model import GAMES_PER_WEEK, CategoryModel, Dist
from app.analysis.perceived import PerceivedValue
from app.analysis.projection import CLOSE
from app.analysis.sim import EACH_CATEGORY, SeasonResult, SeasonSim, simulate_season
from app.analysis.tiers import droppable, has_market, tier
from app.analysis.timing import MatchupTiming, Timing
from app.analysis.trades import SCREEN_MIN_THEIR_GAIN, Deal, Mode, TradeFinder, acceptance_label, mix_shapes
from app.espn.parse import PRO_TEAM_ABBREV, League, Player, Team

WHAT_IF_SIMS = 4000  # what-if gains are paired differences: ~0.5 point noise at 4000, ~1 at 2000
# What the report shows is re-run on more seasons: ~0.3 point noise. A trade that passed its checks on
# 4,000 seasons is checked again here, so one that passed on luck drops out. Costs ~0.17s a move and
# ~120 MB while the report runs (League A, 2026-11-06); 32,000 would need ~390 MB of a 512 MB server.
FINE_SIMS = 12000
SEASON_SIMS = FINE_SIMS
# One report runs at a time: each holds a FINE_SIMS run, and two at once could take a 512 MB server
# down (2026-11-02). On half a CPU two at once would each take twice as long anyway.
REPORT_LOCK = threading.Lock()
TRADE_RESULTS = 10
# Most deals fail the both-teams-gain check early in the season, so the report simulates down
# the finder's queue until the list is full. Each simulation takes ~0.04s locally (more on the
# free plan); these caps bound the run.
TRADE_SIMS = {"fair": 120, "edge": 50, "vision": 50}
# Free agents simulated as keepers. The quick category screen misses some good ones: on League A
# (2026-11-06) going from 8 to 25 found Cam Spencer (+3.9 pt) and John Collins (+3.5 pt) for about 1 s.
PICKUP_SIM_TOP = 25
PICKUP_STREAMS = 3  # at most this many this-week-only streams
KEEPER_MIN_GAIN = 0.02  # expected categories a week a pickup must add to be worth keeping
# Expected categories this matchup a stream must add: about one close category moved by 5 points.
STREAM_MIN_CATS = 0.05
PICKUP_VIA_TRADE = 2  # at most this many "trade X to open a spot" ideas
MIN_SAMPLE_GAMES = 5  # a free agent with no projection or last season needs this many games
LOCKED, LOST = 0.70, 0.30
SUPPORTED_SCORING = {"H2H_CATEGORY", "H2H_MOST_CATEGORIES"}
# Share of a report's run time each step takes, measured on League A (2026-11-07): (stage, start, end).
# Free agents and the season table come first, so they can be shown while trades are screened. The step
# is called "Screening free agents" so an older website, which doesn't know the name, shows it as part of
# loading the league rather than jumping back from trades.
PICKUP_SPAN = ("Screening free agents", 0.04, 0.18)
FAIR_SPAN = ("Screening trades", 0.18, 0.72)
EDGE_SPAN = ("Screening edge trades", 0.72, 0.92)
VISION_SPAN = ("Screening vision trades", 0.92, 0.98)
# The parts of a report a preview leaves for later.
TRADE_SECTIONS = ["trades", "edge_trades", "vision_trades"]
# A vision trade can cost a little now for a stronger playoff run: playoff odds may dip this much.
VISION_ODDS_SLACK = 0.02
# A fair trade raises the other team's playoff odds visibly too: their owner can see it's good for them.
PARTNER_MIN_ODDS_GAIN = 0.01
LOCK_ODDS = 0.98  # a team this sure of the playoffs can't rise a point: it only must not drop
# A trade worth a try (edge lists, Trade for, Trade Park) may instead be a win on paper for them:
# they get at least the trade value they send, and their playoff odds drop by no more than this.
# On League A (2026-11-06) only 2 of ~1,600 deals raised both teams' odds a point; the median deal
# that helps me costs them 4.7 points, so this keeps the deals that cost them least, not lopsided ones.
PAPER_WIN_RATIO = 1.0
PAPER_MAX_PARTNER_DROP = 0.03
# Within a trade step: pairing up players with every team takes about half the time (League A,
# 2026-11-06: 10 s of 17.5 s for fair trades), simulating the best deals the other half.
PAIRING_SHARE = 0.5

Progress = Callable[[str, float], None]
# Projected final places, averaged over simulated seasons: (mine before, after), (theirs before, after).
Finish = tuple[tuple[float, float], tuple[float, float]]


def partner_gains(before: float, after: float) -> bool:
    """Whether the other team's playoff odds go up enough for their owner to see it.

    A team already at (or near) certain gets in can't gain a point, so it only has to not drop.
    """
    if before >= LOCK_ODDS:
        return after >= before
    return after - before >= PARTNER_MIN_ODDS_GAIN - 1e-9


def paper_win(d: Deal) -> bool:
    """Whether the other owner gets at least the trade value they send (owner premium included)."""
    return d.give_value >= PAPER_WIN_RATIO * d.get_value - 1e-9


def worth_a_try(d: Deal, before: float, after: float) -> bool:
    """Whether the other owner has a reason to say yes: their odds go up, or they win on paper
    and their odds drop by no more than PAPER_MAX_PARTNER_DROP."""
    if partner_gains(before, after):
        return True
    return paper_win(d) and after - before >= -PAPER_MAX_PARTNER_DROP - 1e-9


def i_gain(before: float, after: float) -> bool:
    """Whether a trade raises my playoff odds by the same visible point it asks of the other team.

    A trade that leaves me at 52% -> 52% isn't worth making, however it looks in the categories.
    """
    return partner_gains(before, after)


def _round(x: float | None, digits: int = 2) -> float | None:
    return None if x is None else round(x, digits)


def _playoff_payoff(t: schemas.TradeIdea) -> float:
    """Change in playoff odds times expected categories won per playoff week."""
    before = t.playoff_pct_before * (t.playoff_week_cats_before or 0.0)
    return t.playoff_pct_after * (t.playoff_week_cats_after or 0.0) - before


def record(t: Team) -> str:
    return f"{t.wins}-{t.losses}-{t.ties}"


def league_summary(league: League) -> schemas.LeagueSummary:
    supported = league.scoring_type in SUPPORTED_SCORING
    return schemas.LeagueSummary(
        id=league.id,
        season=league.season,
        name=league.name,
        scoring_type=league.scoring_type,
        categories=[c.name for c in league.categories],
        teams=[schemas.TeamSummary(id=t.id, name=t.name, abbrev=t.abbrev, record=record(t)) for t in league.teams],
        playoff_teams=league.playoff_teams,
        is_public=league.is_public,
        supported=supported,
        unsupported_reason=None
        if supported
        else f"{league.scoring_type} leagues are not supported yet. Only head-to-head category leagues are.",
        draft_type="auction" if league.is_auction else "snake",
        round_values=PerceivedValue(league).round_values(),
    )


def same_dist(a: Dist, b: Dist) -> bool:
    return a is b or (np.array_equal(a.mu, b.mu) and np.array_equal(a.sd, b.sd))


class Analyzer:
    def __init__(
        self,
        league: League,
        team_id: int,
        keep_player_ids: list[int] | None = None,
        not_for_sale_player_ids: list[int] | None = None,
        progress: Progress | None = None,
        now: datetime | None = None,
    ):
        self.league = league
        self.now = now or datetime.now(UTC)
        self.progress = progress
        self.model = CategoryModel(league)
        self.teams = league.teams
        self.idx = next(i for i, t in enumerate(self.teams) if t.id == team_id)
        self.me = self.teams[self.idx]
        self.keep = set(keep_player_ids or [])
        self.dists = [self.model.dist(t.roster) for t in self.teams]
        self.cat_names = [c.name for c in self.model.cats]
        self.perceived = PerceivedValue(league, self.model)
        mine = {p.id for p in self.me.roster}
        not_for_sale = {pid for pid in not_for_sale_player_ids or [] if pid not in mine}
        self.premium = OwnerPremium(league, self.model, self.perceived, not_for_sale)
        self._sim: SeasonSim | None = None  # drawn on the first what-if, shared by every later one
        self._fine_sim: SeasonSim | None = None  # the same on FINE_SIMS seasons, for what the report shows
        self._today: tuple[float, np.ndarray] | None = None
        self._league_today: np.ndarray | None = None  # every team's playoff odds today
        self._after_pickup: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
        self._fine_cache: dict[tuple[int, int] | None, SeasonResult] = {}  # the league today / after a pickup
        self._timing: MatchupTiming | None = None

    # ---- helpers -----------------------------------------------------------------------

    def ref(self, p: Player) -> schemas.PlayerRef:
        return schemas.PlayerRef(
            id=p.id,
            name=p.name,
            pro_team=PRO_TEAM_ABBREV.get(p.pro_team_id, ""),
            positions=p.positions,
            injury_status=p.injury_status,
            percent_owned=round(p.percent_owned, 1),
            value=round(self.model.value(p), 2),
            perceived_value=round(self.perceived.value(p), 1),
            tier=tier(p, self.keep),
            badges=badge_refs(self.premium, p),
            owner_premium=round(self.premium.dollars(p), 1),
            games_share=round(p.durability, 3) if p.durability is not None else None,
            injury_prone=injury_prone(p),
            games_out=round(p.games_out, 1) if p.games_out is not None else None,
            expected_back=self.model.back_date(p),
            out_report=p.out_report,
        )

    def _tick(self, span: tuple[str, float, float] | None, done: int, total: int) -> None:
        """Report how far through ``span`` the work is, for callers showing a progress bar."""
        if self.progress is not None and span is not None:
            stage, lo, hi = span
            self.progress(stage, lo + (hi - lo) * (done / total if total else 1.0))

    def avg_cat_win(self, i: int, dists: list[Dist]) -> np.ndarray:
        """(C,) average weekly win probability per category against the rest of the league."""
        return self.model.avg_cat_win(i, dists)

    def period_dists(self, changed: dict[int, list[Player]] | None = None) -> dict[int, list[Dist]] | None:
        """Each team's projection for every remaining matchup period, with changed rosters swapped in."""
        if not self.model.cal:
            return None
        rosters = [(changed or {}).get(i, t.roster) for i, t in enumerate(self.teams)]
        periods = sorted({m.period for m in self.league.schedule if m.winner == "UNDECIDED"})
        return {p: [self.model.dist(r, p) for r in rosters] for p in periods}

    def playoff_periods(self) -> list[int]:
        """The league's playoff matchup periods, from the NBA calendar."""
        if not self.model.cal:
            return []
        return sorted(p for p in self.model.cal.period_days if p > self.league.regular_season_periods)

    def playoff_strength(self, i: int, changed: dict[int, list[Player]] | None = None) -> float | None:
        """Team ``i``'s expected categories won per playoff week against an average team.

        Each playoff week is projected on its real games (who plays, who's back from injury),
        so a player out until then counts fully and a team with few games that week counts less.
        """
        periods = self.playoff_periods()
        if not periods:
            return None
        rosters = [(changed or {}).get(j, t.roster) for j, t in enumerate(self.teams)]
        return float(np.mean([self.model.strength(i, [self.model.dist(r, p) for r in rosters]) for p in periods]))

    def playoff_odds(
        self,
        dists: list[Dist],
        changed: dict[int, list[Player]] | None = None,
        sims: int = WHAT_IF_SIMS,
        week: Dist | None = None,
        fine: bool = False,
    ) -> np.ndarray:
        """Every team's playoff odds with these projections. ``changed`` holds the rosters that differ from today's;
        ``week`` replaces my days left in the current matchup (a move made later this week, or only for it).
        ``fine`` runs it on FINE_SIMS seasons, for odds the report shows."""
        return self.season_result(dists, changed, sims, week, fine).playoff_pct

    def season_result(
        self,
        dists: list[Dist],
        changed: dict[int, list[Player]] | None = None,
        sims: int = WHAT_IF_SIMS,
        week: Dist | None = None,
        fine: bool = False,
    ) -> SeasonResult:
        """The simulated season behind ``playoff_odds``: odds, records and finishing places."""
        period_dists = self.period_dists(changed)
        cur = self.league.current_period
        if week is not None and period_dists and cur in period_dists:
            period_dists[cur] = list(period_dists[cur])
            period_dists[cur][self.idx] = week
        if sims != WHAT_IF_SIMS:
            return simulate_season(self.league, dists, self.model.reverse, n_sims=sims, period_dists=period_dists)
        if fine:
            sim = self.fine_sim()
        else:
            if self._sim is None:
                self._sim = SeasonSim(
                    self.league, self.dists, self.model.reverse, n_sims=WHAT_IF_SIMS, period_dists=self.period_dists()
                )
            sim = self._sim
        moved = set(changed or {}) | {i for i, d in enumerate(dists) if not same_dist(d, self.dists[i])}
        if week is not None:
            moved.add(self.idx)
        return sim.what_if(moved, dists, period_dists if moved else None)

    def fine_sim(self) -> SeasonSim:
        """Today's season on FINE_SIMS seasons, drawn once: shown odds and the season table share it."""
        if self._fine_sim is None:
            self._fine_sim = SeasonSim(
                self.league, self.dists, self.model.reverse, n_sims=FINE_SIMS, period_dists=self.period_dists()
            )
        return self._fine_sim

    def recheck(self, d: Deal) -> tuple[float, float, tuple[float, float], Finish]:
        """A deal's odds on FINE_SIMS seasons: my baseline, mine after it, the partner's before and after,
        and where both teams are projected to finish (mine before and after, then theirs)."""
        key = None if d.vs_pickup is None else (d.vs_pickup[0].id, d.vs_pickup[1].id)
        if None not in self._fine_cache:
            self._fine_cache[None] = self.season_result(self.dists, fine=True)
        if key not in self._fine_cache:
            add, drop = d.vs_pickup
            roster = [p for p in self.me.roster if p.id != drop.id] + [add]
            dists = list(self.dists)
            dists[self.idx] = self.model.dist(roster)
            self._fine_cache[key] = self.season_result(dists, {self.idx: roster}, fine=True)
        base, today, j = self._fine_cache[key], self._fine_cache[None], d.partner_idx
        after = self.season_result(d.dists, d.rosters, fine=True)
        finish = (
            (float(base.mean_rank[self.idx]), float(after.mean_rank[self.idx])),
            (float(today.mean_rank[j]), float(after.mean_rank[j])),
        )
        odds = after.playoff_pct
        mine, theirs = float(base.playoff_pct[self.idx]), (float(today.playoff_pct[j]), float(odds[j]))
        return mine, float(odds[self.idx]), theirs, finish

    def playoff_pct(
        self,
        dists: list[Dist],
        changed: dict[int, list[Player]] | None = None,
        sims: int = WHAT_IF_SIMS,
        week: Dist | None = None,
        fine: bool = False,
    ) -> float:
        """My playoff odds with these projections (see ``playoff_odds``)."""
        return float(self.playoff_odds(dists, changed, sims, week, fine)[self.idx])

    def deal_odds(self, d: Deal) -> tuple[float, tuple[float, float]]:
        """My playoff odds after the deal, and the partner's before and after it."""
        if self._league_today is None:
            self._league_today = self.playoff_odds(self.dists)
        odds = self.playoff_odds(d.dists, d.rosters)
        j = d.partner_idx
        return float(odds[self.idx]), (float(self._league_today[j]), float(odds[j]))

    def baseline(self, d: Deal) -> tuple[float, np.ndarray]:
        """My playoff odds and category win rates that a deal is measured against.

        Today's roster, or my roster after the free pickup when the deal's open spot uses it.
        """
        if d.vs_pickup is None:
            if self._today is None:
                self._today = (self.playoff_pct(self.dists), self.avg_cat_win(self.idx, self.dists))
            return self._today
        add, drop = d.vs_pickup
        key = (add.id, drop.id)
        if key not in self._after_pickup:
            roster = [p for p in self.me.roster if p.id != drop.id] + [add]
            dists = list(self.dists)
            dists[self.idx] = self.model.dist(roster)
            self._after_pickup[key] = (self.playoff_pct(dists, {self.idx: roster}), self.avg_cat_win(self.idx, dists))
        return self._after_pickup[key]

    def trade_idea(
        self,
        finder: TradeFinder,
        d: Deal,
        before: float,
        after: float,
        base_cats: np.ndarray,
        theirs: tuple[float, float] | None = None,
        finish: Finish | None = None,
    ) -> schemas.TradeIdea:
        cats = self.avg_cat_win(self.idx, d.dists) - base_cats
        return schemas.TradeIdea(
            partner_team_id=self.teams[d.partner_idx].id,
            partner_team_name=self.teams[d.partner_idx].name,
            give=[self.ref(p) for p in d.give],
            get=[self.ref(p) for p in d.get],
            my_drop=self.ref(d.my_drop) if d.my_drop else None,
            their_drop=self.ref(d.their_drop) if d.their_drop else None,
            my_add=self.ref(d.my_add) if d.my_add else None,
            my_cats_gain=round(d.my_gain, 3),
            their_cats_gain=round(d.their_gain, 3),
            edge=round(d.edge, 3),
            value_they_get=round(d.give_value, 1),
            value_they_give=round(d.get_value, 1),
            owner_premium=round(d.premium, 1),
            warnings=d.warnings,
            acceptance=round(d.acceptance, 2),
            acceptance_label=acceptance_label(d.acceptance),
            pitch=finder.pitch(d),
            reasons=d.reasons,
            facts=d.facts,
            pitch_needs=finder.pitch_needs(d),
            playoff_pct_before=round(before, 3),
            playoff_pct_after=round(after, 3),
            partner_playoff_pct_before=round(theirs[0], 3) if theirs else None,
            partner_playoff_pct_after=round(theirs[1], 3) if theirs else None,
            playoff_week_cats_before=_round(self.playoff_strength(self.idx)),
            playoff_week_cats_after=_round(self.playoff_strength(self.idx, d.rosters)),
            category_changes={n: round(float(cats[c]), 3) for c, n in enumerate(self.cat_names)},
            vision=d.vision,
            category_detail=self.category_detail(d),
            playoff_weeks=self.playoff_weeks(d),
            finish_before=_round(finish[0][0]) if finish else None,
            finish_after=_round(finish[0][1]) if finish else None,
            partner_finish_before=_round(finish[1][0]) if finish else None,
            partner_finish_after=_round(finish[1][1]) if finish else None,
        )

    def category_detail(self, d: Deal) -> list[schemas.CategoryDetail]:
        """Both teams' weekly projection and weekly win chance in each category, before and after a deal.

        Measured against today's rosters, so it shows what the trade itself changes on each side.
        """
        j = d.partner_idx
        mine = (self.avg_cat_win(self.idx, self.dists), self.avg_cat_win(self.idx, d.dists))
        theirs = (self.avg_cat_win(j, self.dists), self.avg_cat_win(j, d.dists))
        return [
            schemas.CategoryDetail(
                category=n,
                mine_before=round(float(self.dists[self.idx].mu[c]), 4),
                mine_after=round(float(d.dists[self.idx].mu[c]), 4),
                my_win_before=round(float(mine[0][c]), 3),
                my_win_after=round(float(mine[1][c]), 3),
                theirs_before=round(float(self.dists[j].mu[c]), 4),
                theirs_after=round(float(d.dists[j].mu[c]), 4),
                their_win_before=round(float(theirs[0][c]), 3),
                their_win_after=round(float(theirs[1][c]), 3),
            )
            for c, n in enumerate(self.cat_names)
        ]

    def playoff_weeks(self, d: Deal) -> list[schemas.PlayoffWeek]:
        """Each fantasy playoff week on its real schedule: both teams' expected categories won against an
        average team, before and after the deal."""
        cal = self.model.cal
        out = []
        today = [t.roster for t in self.teams]
        after = [d.rosters.get(i, r) for i, r in enumerate(today)]
        for p in self.playoff_periods():
            days = cal.period_days.get(p) or []
            if not days:
                continue
            before_d = [self.model.dist(r, p) for r in today]
            after_d = [self.model.dist(r, p) for r in after]
            out.append(
                schemas.PlayoffWeek(
                    period=p,
                    starts_on=cal.date_of(days[0]).isoformat(),
                    ends_on=cal.date_of(days[-1]).isoformat(),
                    my_cats_before=round(self.model.strength(self.idx, before_d), 2),
                    my_cats_after=round(self.model.strength(self.idx, after_d), 2),
                    their_cats_before=round(self.model.strength(d.partner_idx, before_d), 2),
                    their_cats_after=round(self.model.strength(d.partner_idx, after_d), 2),
                )
            )
        return out

    # ---- sections ----------------------------------------------------------------------

    def categories(self) -> list[schemas.CategoryOutlook]:
        win = self.avg_cat_win(self.idx, self.dists)
        mus = np.array([d.mu for d in self.dists])
        signed = np.where(self.model.reverse, -mus, mus)
        out = []
        for c, name in enumerate(self.cat_names):
            rank = int((signed[:, c] > signed[self.idx, c]).sum()) + 1
            p = float(win[c])
            out.append(
                schemas.CategoryOutlook(
                    category=name,
                    win_pct=round(p, 3),
                    league_rank=rank,
                    weekly_projection=round(float(mus[self.idx, c]), 3),
                    tag="locked" if p >= LOCKED else "lost" if p <= LOST else "swing",
                )
            )
        return out

    def builds(self) -> list[schemas.TeamBuild]:
        mus = np.array([d.mu for d in self.dists])
        signed = np.where(self.model.reverse, -mus, mus)
        z = (signed - signed.mean(0)) / (signed.std(0) + 1e-9)
        out = []
        for i, t in enumerate(self.teams):
            ranks = {n: int((signed[:, c] > signed[i, c]).sum()) + 1 for c, n in enumerate(self.cat_names)}
            out.append(
                schemas.TeamBuild(
                    team_id=t.id,
                    team_name=t.name,
                    ranks=ranks,
                    strong=[n for c, n in enumerate(self.cat_names) if z[i, c] >= 1.0],
                    weak=[n for c, n in enumerate(self.cat_names) if z[i, c] <= -1.0],
                    expected_cats_won=round(self.model.strength(i, self.dists), 2),
                )
            )
        return out

    def trades(self, mode: Mode = "fair", span: tuple[str, float, float] | None = None) -> list[schemas.TradeIdea]:
        self._tick(span, 0, 1)
        # The playoff-odds check below has the final say, so the screen lets through deals that leave
        # their average week flat: their odds can still rise (healthy players for injured ones).
        finder = TradeFinder(
            self.league,
            self.model,
            self.idx,
            self.keep,
            mode,
            premium=self.premium,
            min_their_gain=SCREEN_MIN_THEIR_GAIN,
        )
        ideas = []
        picks = finder.picks(TRADE_RESULTS)
        budget = TRADE_SIMS[mode]
        sims = 0
        pairing = sim_span = None
        if span is not None:
            stage, lo, hi = span
            mid = lo + (hi - lo) * PAIRING_SHARE
            pairing, sim_span = (stage, lo, mid), (stage, mid, hi)
        queue = finder.queue(on_partner=lambda done, total: self._tick(pairing, done, total))
        for d in queue:
            if picks.full or sims >= budget:
                break
            if not picks.room(d):
                continue
            self._tick(sim_span, sims, budget)
            sims += 1
            before, base_cats = self.baseline(d)
            after, theirs = self.deal_odds(d)
            if not self._holds_up(mode, d, before, after, theirs):
                continue
            # Checked again on more seasons, which also gives the odds the list shows.
            before, after, theirs, finish = self.recheck(d)
            if not self._holds_up(mode, d, before, after, theirs):
                continue
            picks.add(d)
            ideas.append(self.trade_idea(finder, d, before, after, base_cats, theirs, finish))
        if mode == "vision":
            # Expected playoff payoff: the chance to be there times how strong I am once there.
            ideas.sort(key=lambda t: t.acceptance * _playoff_payoff(t), reverse=True)
        else:
            # Expected payoff: playoff gain weighted by the chance the deal actually happens.
            ideas.sort(key=lambda t: t.acceptance * (t.playoff_pct_after - t.playoff_pct_before), reverse=True)
        return mix_shapes(ideas, TRADE_RESULTS, lambda t: len(t.give) != len(t.get))

    @staticmethod
    def _holds_up(mode: Mode, d: Deal, before: float, after: float, theirs: tuple[float, float]) -> bool:
        """Small category gains can vanish in the full simulation; only keep moves that hold up there.

        A vision trade may cost a little now for a stronger playoff run; any other trade must raise my
        odds by a point, as it must theirs. Fair trades help both teams; edge trades may instead be a
        paper win for them that costs them little.
        """
        if mode == "vision":
            if after < before - VISION_ODDS_SLACK:
                return False
        elif not i_gain(before, after):
            return False
        return worth_a_try(d, *theirs) if mode == "edge" else partner_gains(*theirs)

    def matchup_timing(self) -> MatchupTiming:
        if self._timing is None:
            self._timing = MatchupTiming(self.league, self.me.id, self.model, int(self.now.timestamp() * 1000))
        return self._timing

    def adds_left(self) -> int | None:
        """Pickups you can still make in this matchup; None when the league has no limit."""
        period = self.league.current_period
        allowed = self.league.adds_allowed(period)
        if allowed is None:
            return None
        return max(allowed - self.me.period_adds.get(period, 0), 0)

    def week_matchup(self) -> schemas.WeekMatchup | None:
        t = self.matchup_timing()
        if not t.ready:
            return None
        return schemas.WeekMatchup(
            opponent_team_id=t.opponent.id,
            opponent_name=t.opponent.name,
            adds_used=self.me.period_adds.get(self.league.current_period, 0),
            adds_allowed=self.league.adds_allowed(self.league.current_period),
            close_categories=[n for c, n in enumerate(self.cat_names) if abs(float(t.base[c]) - 0.5) <= CLOSE],
        )

    def _weeks_after_this(self) -> float:
        """Weeks of the regular season left after the current matchup."""
        if not self.model.cal or "season" not in self.model.horizon:
            return 0.0
        season_days, _ = self.model.horizon["season"]
        week_days, _ = self.model.horizon.get(self.league.current_period, ([], 1.0))
        return max(len(season_days) - len(week_days), 0) / 7

    def pickups(self, span: tuple[str, float, float] | None = None) -> list[schemas.PickupIdea]:
        """Free agents worth adding: keepers that help the rest of the season, and streams that
        only help this week's matchup (more games, or a category close enough to steal).

        A keeper is ranked on the categories it adds over the rest of the season plus this
        week, made on its best day this week. A stream is a swap for this week only: the
        player you drop has no trade value, so another like him is there next week, and only
        this matchup is counted.
        """
        self._tick(span, 0, 1)
        base = self.model.strength(self.idx, self.dists)
        base_cats = self.avg_cat_win(self.idx, self.dists)
        timing = self.matchup_timing()
        left = self.adds_left()
        can_add = left is None or left > 0
        weeks_after = self._weeks_after_this()
        # Only players with no trade value are ever released. A swap that needs a
        # player with a market is shown as "trade him to open the spot" instead.
        outgoing = [p for p in self.me.roster if p.id not in self.keep and p.available]
        candidates = []
        streams: list[tuple[Timing, Player, Player]] = []
        for fa in self.league.free_agents:
            # No NBA team (unsigned) means no games, whatever his stats say.
            if not fa.available or not fa.pro_team_id or (not fa.has_prior and fa.games_played < MIN_SAMPLE_GAMES):
                continue
            best: dict[bool, tuple] = {}
            stream: tuple[Timing, Player, Player] | None = None
            for drop in outgoing:
                roster = [p for p in self.me.roster if p.id != drop.id] + [fa]
                dists = list(self.dists)
                dists[self.idx] = self.model.dist(roster)
                gain = self.model.strength(self.idx, dists) - base
                via_trade = has_market(drop)
                if via_trade not in best or gain > best[via_trade][0]:
                    best[via_trade] = (gain, drop, dists, roster)
                if not via_trade and can_add and gain <= KEEPER_MIN_GAIN:
                    t = timing.best(fa, drop)
                    better = t is not None and (stream is None or t.cats_best > stream[0].cats_best)
                    if better and t.cats_best >= STREAM_MIN_CATS:
                        stream = (t, fa, drop)
            for via_trade, (gain, drop, dists, roster) in best.items():
                if gain > KEEPER_MIN_GAIN:
                    t = timing.best(fa, drop) if not via_trade and can_add else None
                    candidates.append((gain, fa, drop, dists, via_trade, roster, t))
            if stream is not None:
                streams.append(stream)

        def total(c: tuple) -> float:
            """Expected categories the move adds: every week after this one, plus this one."""
            gain, t = c[0], c[6]
            return gain * weeks_after + (t.cats_best if t is not None else gain)

        released = sorted((c for c in candidates if not c[4]), key=total, reverse=True)[:PICKUP_SIM_TOP]
        traded = sorted((c for c in candidates if c[4]), key=lambda c: c[0], reverse=True)
        # A trade-first idea is worth showing only if it beats every straight pickup.
        top_release = max((c[0] for c in released), default=0.0)
        traded = [c for c in traded if c[0] > top_release + 0.05][:PICKUP_VIA_TRADE]
        streams = sorted(streams, key=lambda s: s[0].cats_best, reverse=True)[:PICKUP_STREAMS]
        before = self.playoff_pct(self.dists, fine=True)
        # With no pickups left this matchup, a move only counts from next week.
        this_week = None if can_add else self.model.dist(self.me.roster, self.league.current_period)
        ideas = []
        to_sim = len(released) + len(traded) + len(streams)
        for n, (gain, fa, drop, dists, via_trade, roster, t) in enumerate(released + traded):
            self._tick(span, n, to_sim)
            cats = self.avg_cat_win(self.idx, dists) - base_cats
            week = t.rest if t is not None else (None if via_trade else this_week)
            ideas.append(
                schemas.PickupIdea(
                    add=self.ref(fa),
                    drop=self.ref(drop),
                    open_spot_by_trade=via_trade,
                    availability=fa.status,
                    my_cats_gain=round(gain, 3),
                    playoff_pct_before=round(before, 3),
                    playoff_pct_after=round(self.playoff_pct(dists, {self.idx: roster}, week=week, fine=True), 3),
                    category_changes={n: round(float(cats[c]), 3) for c, n in enumerate(self.cat_names)},
                    timing=self.move_timing(t),
                )
            )
        for n, (t, fa, drop) in enumerate(streams, start=len(released) + len(traded)):
            self._tick(span, n, to_sim)
            ideas.append(
                schemas.PickupIdea(
                    add=self.ref(fa),
                    drop=self.ref(drop),
                    open_spot_by_trade=False,
                    stream=True,
                    availability=fa.status,
                    my_cats_gain=round(t.cats_best, 3),
                    playoff_pct_before=round(before, 3),
                    playoff_pct_after=round(self.playoff_pct(self.dists, week=t.rest, fine=True), 3),
                    category_changes={n: round(float(t.cat_changes[c]), 3) for c, n in enumerate(self.cat_names)},
                    timing=self.move_timing(t),
                )
            )
        ideas = [p for p in ideas if p.playoff_pct_after >= p.playoff_pct_before]
        ideas.sort(key=lambda p: (not p.open_spot_by_trade, p.playoff_pct_after, p.my_cats_gain), reverse=True)
        return ideas

    def move_timing(self, t: Timing | None) -> schemas.MoveTiming | None:
        if t is None:
            return None
        cal = self.model.cal
        last = self.model.horizon[self.league.current_period][0][-1]
        wait = t.best_day != t.earliest_day
        return schemas.MoveTiming(
            wait=wait,
            make_on=cal.date_of(t.best_day if wait else cal.today).isoformat(),
            next_week=t.best_day > last,
            earliest_on=cal.date_of(t.earliest_day).isoformat(),
            cats_now=round(t.cats_earliest, 3),
            cats_best=round(t.cats_best, 3),
            drop_games_kept=t.drop_games_kept,
            add_games_missed=t.add_games_missed,
            week_changes={n: round(float(t.cat_changes[c]), 3) for c, n in enumerate(self.cat_names)},
            days=[
                schemas.MoveDay(date=cal.date_of(d).isoformat(), add_plays=a, drop_plays=p)
                for d, a, p in zip(t.days, t.add_plays, t.drop_plays, strict=True)
            ],
        )

    def season(self) -> list[schemas.SeasonRow]:
        res = self.fine_sim().base
        rows = [
            schemas.SeasonRow(
                team_id=t.id,
                team_name=t.name,
                current_record=record(t),
                projected_wins=round(float(res.wins[i]), 1),
                projected_losses=round(float(res.losses[i]), 1),
                projected_ties=round(float(res.ties[i]), 1),
                playoff_pct=round(float(res.playoff_pct[i]), 3),
                first_seed_pct=round(float(res.first_seed_pct[i]), 3),
                mean_rank=round(float(res.mean_rank[i]), 2),
                playoff_week_cats=_round(self.playoff_strength(i)),
            )
            for i, t in enumerate(self.teams)
        ]
        return sorted(rows, key=lambda r: r.mean_rank)

    def league_rosters(self) -> list[schemas.TeamRoster]:
        """Other teams' players with how much each would help you, for the trade-for picker."""
        cuts = droppable(self.me.roster, self.keep)
        cut = min(cuts, key=self.model.value) if cuts else None
        mine = [p for p in self.me.roster if p is not cut]
        base = self.model.strength(self.idx, self.dists)
        out = []
        for i, t in enumerate(self.teams):
            if i == self.idx:
                continue
            players = []
            for p in t.roster:
                dists = list(self.dists)
                dists[self.idx] = self.model.dist(mine + [p])
                gain = self.model.strength(self.idx, dists) - base
                players.append(schemas.TradeTarget(player=self.ref(p), my_cats_gain=round(gain, 3)))
            players.sort(key=lambda x: x.player.perceived_value, reverse=True)
            out.append(schemas.TeamRoster(team_id=t.id, team_name=t.name, players=players))
        return out

    def report(self, on_preview: Callable[[schemas.AnalysisReport], None] | None = None) -> schemas.AnalysisReport:
        """The whole analysis. ``on_preview`` gets the report early, once everything but the trade lists is
        in (``pending`` names what's missing), so a caller can show free agents while trades are screened."""
        if not REPORT_LOCK.acquire(blocking=False):
            if self.progress is not None:
                self.progress("Waiting for another analysis", 0.02)
            REPORT_LOCK.acquire()
        try:
            return self._report(on_preview)
        finally:
            self._fine_sim = None  # the big run's memory goes back before the next report starts
            REPORT_LOCK.release()

    def _report(self, on_preview: Callable[[schemas.AnalysisReport], None] | None) -> schemas.AnalysisReport:
        started = time.perf_counter()
        each_cat = self.league.scoring_type == EACH_CATEGORY
        roster = sorted(self.me.roster, key=self.model.value, reverse=True)
        report = schemas.AnalysisReport(
            league=league_summary(self.league),
            team_id=self.me.id,
            team_name=self.me.name,
            roster=[self.ref(p) for p in roster],
            categories=self.categories(),
            builds=self.builds(),
            trades=[],
            edge_trades=[],
            vision_trades=[],
            pickups=self.pickups(span=PICKUP_SPAN),
            season=self.season(),
            pending=list(TRADE_SECTIONS),
            league_rosters=self.league_rosters(),
            faab_remaining=self.me.faab_remaining,
            week=self.week_matchup(),
            waiver_priority=self.me.waiver_priority,
            assumptions=schemas.Assumptions(
                games_per_week=GAMES_PER_WEEK,
                simulations=SEASON_SIMS,
                scoring_type=self.league.scoring_type,
                notes=[
                    "Projections are ESPN's season projections, blended with this season's stats as games are played.",
                    f"Each team starts its top {self.model.n_starters} healthy players by value, "
                    f"about {GAMES_PER_WEEK} games each per week.",
                    "Players listed OUT or on IR are left out until they return.",
                    "Every category counts as a win or loss in the standings."
                    if each_cat
                    else "Each matchup counts as one win, loss or tie in the standings.",
                    "Fair trades are shown only if they raise both teams' playoff odds. Edge trades raise yours "
                    "and give the other team at least equal trade value, costing its playoff odds at most "
                    f"{round(PAPER_MAX_PARTNER_DROP * 100)} points. Trade value starts from ESPN "
                    "rankings and draft prices (draft positions in snake leagues), which fade as games are "
                    "played, and blends % owned, "
                    "rest-of-season value in your categories, this season's rank and the last 15 days. "
                    "Only lightly owned players are ever suggested as drops. One star counts for more "
                    "than two players adding up to the same dollars, and a player they'd have to cut counts "
                    "against the deal. Acceptance odds are a rough estimate, not a prediction.",
                    "Owners want a premium for players they hold on to: jewels (star names), keystones "
                    "(players who carry a category for their team) and fresh signings. Your own jewels and "
                    "keystones are only offered in deals near the top of what the finder sees.",
                ],
            ),
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            elapsed_ms=0,
        )
        if on_preview is not None:
            on_preview(report.model_copy(update={"elapsed_ms": int((time.perf_counter() - started) * 1000)}))
        report.trades = self.trades(span=FAIR_SPAN)
        report.edge_trades = self.trades("edge", span=EDGE_SPAN)
        report.vision_trades = self.trades("vision", span=VISION_SPAN)
        report.pending = []
        report.generated_at = datetime.now(UTC).isoformat(timespec="seconds")
        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return report

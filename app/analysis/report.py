"""Builds the full analysis for one team: category outlook, builds, trades, pickups, season."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np

from app import schemas
from app.analysis.badges import OwnerPremium, badge_refs
from app.analysis.durability import injury_prone
from app.analysis.model import GAMES_PER_WEEK, CategoryModel, Dist
from app.analysis.perceived import PerceivedValue
from app.analysis.sim import EACH_CATEGORY, SeasonSim, simulate_season
from app.analysis.tiers import droppable, has_market, tier
from app.analysis.timing import MatchupTiming, Timing
from app.analysis.trades import SCREEN_MIN_THEIR_GAIN, Deal, Mode, TradeFinder, acceptance_label, mix_shapes
from app.espn.parse import PRO_TEAM_ABBREV, League, Player, Team

SEASON_SIMS = 4000
WHAT_IF_SIMS = 4000  # what-if gains are paired differences: ~0.5 point noise at 4000, ~1 at 2000
TRADE_RESULTS = 10
# Most deals fail the both-teams-gain check early in the season, so the report simulates down
# the finder's queue until the list is full. Each simulation takes ~0.04s locally (more on the
# free plan); these caps bound the run.
TRADE_SIMS = {"fair": 120, "edge": 50, "vision": 50}
PICKUP_SIM_TOP = 8
PICKUP_VIA_TRADE = 2  # at most this many "trade X to open a spot" ideas
MIN_SAMPLE_GAMES = 5  # a free agent with no projection or last season needs this many games
LOCKED, LOST = 0.70, 0.30
SUPPORTED_SCORING = {"H2H_CATEGORY", "H2H_MOST_CATEGORIES"}
# Share of a report's run time each step takes, measured on League A: (stage, start, end).
FAIR_SPAN = ("Screening trades", 0.05, 0.50)
EDGE_SPAN = ("Screening edge trades", 0.50, 0.65)
VISION_SPAN = ("Screening vision trades", 0.65, 0.80)
# A vision trade can cost a little now for a stronger playoff run: playoff odds may dip this much.
VISION_ODDS_SLACK = 0.02
# No trade is suggested unless the other team's playoff odds visibly go up too: the deal has to
# be one their owner can see is good for them, never one that only looks good on paper.
PARTNER_MIN_ODDS_GAIN = 0.01
LOCK_ODDS = 0.98  # a team this sure of the playoffs can't rise a point: it only must not drop
PICKUP_SPAN = ("Screening pickups", 0.80, 0.94)

Progress = Callable[[str, float], None]


def partner_gains(before: float, after: float) -> bool:
    """Whether the other team's playoff odds go up enough for their owner to see it.

    A team already at (or near) certain gets in can't gain a point, so it only has to not drop.
    """
    if before >= LOCK_ODDS:
        return after >= before
    return after - before >= PARTNER_MIN_ODDS_GAIN - 1e-9


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
        self._today: tuple[float, np.ndarray] | None = None
        self._league_today: np.ndarray | None = None  # every team's playoff odds today
        self._after_pickup: dict[tuple[int, int], tuple[float, np.ndarray]] = {}

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
        self, dists: list[Dist], changed: dict[int, list[Player]] | None = None, sims: int = WHAT_IF_SIMS
    ) -> np.ndarray:
        """Every team's playoff odds with these projections. ``changed`` holds the rosters that differ from today's."""
        if sims != WHAT_IF_SIMS:
            res = simulate_season(
                self.league, dists, self.model.reverse, n_sims=sims, period_dists=self.period_dists(changed)
            )
            return res.playoff_pct
        if self._sim is None:
            self._sim = SeasonSim(
                self.league, self.dists, self.model.reverse, n_sims=WHAT_IF_SIMS, period_dists=self.period_dists()
            )
        moved = set(changed or {}) | {i for i, d in enumerate(dists) if not same_dist(d, self.dists[i])}
        period_dists = self.period_dists(changed) if moved else None
        return self._sim.what_if(moved, dists, period_dists).playoff_pct

    def playoff_pct(
        self, dists: list[Dist], changed: dict[int, list[Player]] | None = None, sims: int = WHAT_IF_SIMS
    ) -> float:
        """My playoff odds with these projections. ``changed`` holds the rosters that differ from today's."""
        return float(self.playoff_odds(dists, changed, sims)[self.idx])

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
        )

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
        for d in finder.queue():
            if picks.full or sims >= budget:
                break
            if not picks.room(d):
                continue
            self._tick(span, sims, budget)
            sims += 1
            before, base_cats = self.baseline(d)
            after, theirs = self.deal_odds(d)
            # Small category gains can vanish in the full simulation; only keep moves that hold up there.
            # A vision trade may cost a little now for a stronger playoff run; any other trade must
            # raise my odds by a point, as it must theirs.
            if mode == "vision":
                if after < before - VISION_ODDS_SLACK:
                    continue
            elif not i_gain(before, after):
                continue
            if not partner_gains(*theirs):
                continue
            picks.add(d)
            ideas.append(self.trade_idea(finder, d, before, after, base_cats, theirs))
        if mode == "vision":
            # Expected playoff payoff: the chance to be there times how strong I am once there.
            ideas.sort(key=lambda t: t.acceptance * _playoff_payoff(t), reverse=True)
        else:
            # Expected payoff: playoff gain weighted by the chance the deal actually happens.
            ideas.sort(key=lambda t: t.acceptance * (t.playoff_pct_after - t.playoff_pct_before), reverse=True)
        return mix_shapes(ideas, TRADE_RESULTS, lambda t: len(t.give) != len(t.get))

    def pickups(self, span: tuple[str, float, float] | None = None) -> list[schemas.PickupIdea]:
        self._tick(span, 0, 1)
        base = self.model.strength(self.idx, self.dists)
        base_cats = self.avg_cat_win(self.idx, self.dists)
        # Only players with no trade value are ever released. A swap that needs a
        # player with a market is shown as "trade him to open the spot" instead.
        outgoing = [p for p in self.me.roster if p.id not in self.keep and p.available]
        candidates = []
        for fa in self.league.free_agents:
            # No NBA team (unsigned) means no games, whatever his stats say.
            if not fa.available or not fa.pro_team_id or (not fa.has_prior and fa.games_played < MIN_SAMPLE_GAMES):
                continue
            best: dict[bool, tuple] = {}
            for drop in outgoing:
                roster = [p for p in self.me.roster if p.id != drop.id] + [fa]
                dists = list(self.dists)
                dists[self.idx] = self.model.dist(roster)
                gain = self.model.strength(self.idx, dists) - base
                via_trade = has_market(drop)
                if via_trade not in best or gain > best[via_trade][0]:
                    best[via_trade] = (gain, drop, dists, roster)
            for via_trade, (gain, drop, dists, roster) in best.items():
                if gain > 0.02:
                    candidates.append((gain, fa, drop, dists, via_trade, roster))

        candidates.sort(key=lambda c: c[0], reverse=True)
        released = [c for c in candidates if not c[4]][:PICKUP_SIM_TOP]
        traded = [c for c in candidates if c[4]]
        # A trade-first idea is worth showing only if it beats every straight pickup.
        top_release = released[0][0] if released else 0.0
        traded = [c for c in traded if c[0] > top_release + 0.05][:PICKUP_VIA_TRADE]
        before = self.playoff_pct(self.dists)
        timing = MatchupTiming(self.league, self.me.id, self.model, int(self.now.timestamp() * 1000))
        ideas = []
        to_sim = released + traded
        for n, (gain, fa, drop, dists, via_trade, roster) in enumerate(to_sim):
            self._tick(span, n, len(to_sim))
            cats = self.avg_cat_win(self.idx, dists) - base_cats
            ideas.append(
                schemas.PickupIdea(
                    add=self.ref(fa),
                    drop=self.ref(drop),
                    open_spot_by_trade=via_trade,
                    availability=fa.status,
                    my_cats_gain=round(gain, 3),
                    playoff_pct_before=round(before, 3),
                    playoff_pct_after=round(self.playoff_pct(dists, {self.idx: roster}), 3),
                    category_changes={n: round(float(cats[c]), 3) for c, n in enumerate(self.cat_names)},
                    timing=self.move_timing(timing.best(fa, drop)),
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
        )

    def season(self) -> list[schemas.SeasonRow]:
        res = simulate_season(
            self.league, self.dists, self.model.reverse, n_sims=SEASON_SIMS, period_dists=self.period_dists()
        )
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

    def _season_step(self) -> list[schemas.SeasonRow]:
        if self.progress is not None:
            self.progress("Simulating the season", PICKUP_SPAN[2])
        return self.season()

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

    def report(self) -> schemas.AnalysisReport:
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
            trades=self.trades(span=FAIR_SPAN),
            edge_trades=self.trades("edge", span=EDGE_SPAN),
            vision_trades=self.trades("vision", span=VISION_SPAN),
            pickups=self.pickups(span=PICKUP_SPAN),
            season=self._season_step(),
            league_rosters=self.league_rosters(),
            faab_remaining=self.me.faab_remaining,
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
                    "Trades are shown only if they also help the other team in the categories it competes in "
                    "(its punts are ignored) and give it close to equal trade value. Trade value starts from ESPN "
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
        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return report

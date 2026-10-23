"""Builds the full analysis for one team: category outlook, builds, trades, pickups, season."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import numpy as np

from app import schemas
from app.analysis.model import GAMES_PER_WEEK, CategoryModel, Dist
from app.analysis.perceived import PerceivedValue
from app.analysis.sim import EACH_CATEGORY, simulate_season
from app.analysis.tiers import has_market, tier
from app.analysis.trades import Mode, TradeFinder, acceptance_label
from app.espn.parse import League, Player, Team

SEASON_SIMS = 4000
WHAT_IF_SIMS = 2000
TRADE_RESULTS = 10
PICKUP_SIM_TOP = 8
PICKUP_VIA_TRADE = 2  # at most this many "trade X to open a spot" ideas
MIN_SAMPLE_GAMES = 5  # a free agent with no projection or last season needs this many games
LOCKED, LOST = 0.70, 0.30
SUPPORTED_SCORING = {"H2H_CATEGORY", "H2H_MOST_CATEGORIES"}


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
    )


class Analyzer:
    def __init__(self, league: League, team_id: int, keep_player_ids: list[int] | None = None):
        self.league = league
        self.model = CategoryModel(league)
        self.teams = league.teams
        self.idx = next(i for i, t in enumerate(self.teams) if t.id == team_id)
        self.me = self.teams[self.idx]
        self.keep = set(keep_player_ids or [])
        self.dists = [self.model.dist(t.roster) for t in self.teams]
        self.cat_names = [c.name for c in self.model.cats]
        self.perceived = PerceivedValue(league, self.model)

    # ---- helpers -----------------------------------------------------------------------

    def ref(self, p: Player) -> schemas.PlayerRef:
        return schemas.PlayerRef(
            id=p.id,
            name=p.name,
            positions=p.positions,
            injury_status=p.injury_status,
            percent_owned=round(p.percent_owned, 1),
            value=round(self.model.value(p), 2),
            perceived_value=round(self.perceived.value(p), 1),
            tier=tier(p, self.keep),
        )

    def avg_cat_win(self, i: int, dists: list[Dist]) -> np.ndarray:
        """(C,) average weekly win probability per category against the rest of the league."""
        return self.model.avg_cat_win(i, dists)

    def playoff_pct(self, dists: list[Dist], sims: int = WHAT_IF_SIMS) -> float:
        res = simulate_season(self.league, dists, self.model.reverse, n_sims=sims)
        return float(res.playoff_pct[self.idx])

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

    def trades(self, mode: Mode = "fair") -> list[schemas.TradeIdea]:
        finder = TradeFinder(self.league, self.model, self.idx, self.keep, mode)
        base_cats = self.avg_cat_win(self.idx, self.dists)
        before = self.playoff_pct(self.dists)
        ideas = []
        for d in finder.shortlist():
            after = self.playoff_pct(d.dists)
            # Small category gains can vanish in the full simulation; only keep moves that hold up there.
            if after < before:
                continue
            cats = self.avg_cat_win(self.idx, d.dists) - base_cats
            ideas.append(
                schemas.TradeIdea(
                    partner_team_id=self.teams[d.partner_idx].id,
                    partner_team_name=self.teams[d.partner_idx].name,
                    give=[self.ref(p) for p in d.give],
                    get=[self.ref(p) for p in d.get],
                    my_drop=self.ref(d.my_drop) if d.my_drop else None,
                    their_drop=self.ref(d.their_drop) if d.their_drop else None,
                    my_cats_gain=round(d.my_gain, 3),
                    their_cats_gain=round(d.their_gain, 3),
                    edge=round(d.edge, 3),
                    value_they_get=round(d.give_value, 1),
                    value_they_give=round(d.get_value, 1),
                    acceptance=round(d.acceptance, 2),
                    acceptance_label=acceptance_label(d.acceptance),
                    pitch=finder.pitch(d),
                    reasons=d.reasons,
                    playoff_pct_before=round(before, 3),
                    playoff_pct_after=round(after, 3),
                    category_changes={n: round(float(cats[c]), 3) for c, n in enumerate(self.cat_names)},
                )
            )
        # Expected payoff: playoff gain weighted by the chance the deal actually happens.
        ideas.sort(key=lambda t: t.acceptance * (t.playoff_pct_after - t.playoff_pct_before), reverse=True)
        return ideas[:TRADE_RESULTS]

    def pickups(self) -> list[schemas.PickupIdea]:
        base = self.model.strength(self.idx, self.dists)
        base_cats = self.avg_cat_win(self.idx, self.dists)
        # Only players with no trade value are ever released. A swap that needs a
        # player with a market is shown as "trade him to open the spot" instead.
        outgoing = [p for p in self.me.roster if p.id not in self.keep and p.available]
        candidates = []
        for fa in self.league.free_agents:
            if not fa.available or (not fa.has_prior and fa.games_played < MIN_SAMPLE_GAMES):
                continue
            best: dict[bool, tuple] = {}
            for drop in outgoing:
                roster = [p for p in self.me.roster if p.id != drop.id] + [fa]
                dists = list(self.dists)
                dists[self.idx] = self.model.dist(roster)
                gain = self.model.strength(self.idx, dists) - base
                via_trade = has_market(drop)
                if via_trade not in best or gain > best[via_trade][0]:
                    best[via_trade] = (gain, drop, dists)
            for via_trade, (gain, drop, dists) in best.items():
                if gain > 0.02:
                    candidates.append((gain, fa, drop, dists, via_trade))

        candidates.sort(key=lambda c: c[0], reverse=True)
        released = [c for c in candidates if not c[4]][:PICKUP_SIM_TOP]
        traded = [c for c in candidates if c[4]]
        # A trade-first idea is worth showing only if it beats every straight pickup.
        top_release = released[0][0] if released else 0.0
        traded = [c for c in traded if c[0] > top_release + 0.05][:PICKUP_VIA_TRADE]
        before = self.playoff_pct(self.dists)
        ideas = []
        for gain, fa, drop, dists, via_trade in released + traded:
            cats = self.avg_cat_win(self.idx, dists) - base_cats
            ideas.append(
                schemas.PickupIdea(
                    add=self.ref(fa),
                    drop=self.ref(drop),
                    open_spot_by_trade=via_trade,
                    availability=fa.status,
                    my_cats_gain=round(gain, 3),
                    playoff_pct_before=round(before, 3),
                    playoff_pct_after=round(self.playoff_pct(dists), 3),
                    category_changes={n: round(float(cats[c]), 3) for c, n in enumerate(self.cat_names)},
                )
            )
        ideas = [p for p in ideas if p.playoff_pct_after >= p.playoff_pct_before]
        ideas.sort(key=lambda p: (not p.open_spot_by_trade, p.playoff_pct_after, p.my_cats_gain), reverse=True)
        return ideas

    def season(self) -> list[schemas.SeasonRow]:
        res = simulate_season(self.league, self.dists, self.model.reverse, n_sims=SEASON_SIMS)
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
            )
            for i, t in enumerate(self.teams)
        ]
        return sorted(rows, key=lambda r: r.mean_rank)

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
            trades=self.trades(),
            edge_trades=self.trades("edge"),
            pickups=self.pickups(),
            season=self.season(),
            faab_remaining=self.me.faab_remaining,
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
                    "rankings and draft prices, which fade as games are played, and blends % owned, "
                    "rest-of-season value in your categories, this season's rank and the last 15 days. "
                    "Only lightly owned players are ever suggested as drops. One star counts for more "
                    "than two players adding up to the same dollars, and a player they'd have to cut counts "
                    "against the deal. Acceptance odds are a rough estimate, not a prediction.",
                ],
            ),
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            elapsed_ms=0,
        )
        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return report

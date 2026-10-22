"""Builds the full analysis for one team: category outlook, builds, trades, pickups, season."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import numpy as np

from app import schemas
from app.analysis.model import GAMES_PER_WEEK, CategoryModel, Dist
from app.analysis.sim import EACH_CATEGORY, simulate_season
from app.espn.parse import League, Player, Team

SEASON_SIMS = 4000
WHAT_IF_SIMS = 2000
TRADE_SIM_TOP = 12
PICKUP_SIM_TOP = 8
LOCKED, LOST = 0.70, 0.30
# A partner is assumed to accept only if the deal does not make them worse.
PARTNER_TOLERANCE = -0.02
# Owners judge trades on generic value. Deals that look this lopsided against them are skipped.
MAX_PAPER_GAP = 2.5
EVEN_PAPER_GAP = 0.5
MAX_IDEAS_PER_PLAYER = 2
MAX_IDEAS_PER_PARTNER = 4
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

    # ---- helpers -----------------------------------------------------------------------

    def ref(self, p: Player) -> schemas.PlayerRef:
        return schemas.PlayerRef(
            id=p.id,
            name=p.name,
            positions=p.positions,
            injury_status=p.injury_status,
            percent_owned=round(p.percent_owned, 1),
            value=round(self.model.value(p), 2),
        )

    def avg_cat_win(self, i: int, dists: list[Dist]) -> np.ndarray:
        """(C,) average weekly win probability per category against the rest of the league."""
        return np.mean([self.model.cat_win_prob(dists[i], d) for j, d in enumerate(dists) if j != i], axis=0)

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

    def trades(self) -> list[schemas.TradeIdea]:
        base_strength = [self.model.strength(i, self.dists) for i in range(len(self.teams))]
        base_cats = self.avg_cat_win(self.idx, self.dists)
        candidates = []
        mine = [p for p in self.me.roster if p.id not in self.keep]
        for j, partner in enumerate(self.teams):
            if j == self.idx:
                continue
            for give in mine:
                for get in partner.roster:
                    if self.model.value(get) - self.model.value(give) > MAX_PAPER_GAP:
                        continue
                    my_roster = [p for p in self.me.roster if p.id != give.id] + [get]
                    their_roster = [p for p in partner.roster if p.id != get.id] + [give]
                    dists = list(self.dists)
                    dists[self.idx] = self.model.dist(my_roster)
                    dists[j] = self.model.dist(their_roster)
                    my_gain = self.model.strength(self.idx, dists) - base_strength[self.idx]
                    their_gain = self.model.strength(j, dists) - base_strength[j]
                    if my_gain <= 0.05 or their_gain < PARTNER_TOLERANCE:
                        continue
                    candidates.append((my_gain, their_gain, j, give, get, dists))

        # Best gain first; among equal gains, give up the least valuable player.
        candidates.sort(key=lambda c: (round(c[0], 3), -self.model.value(c[3])), reverse=True)
        shortlist, per_player, per_partner = [], {}, {}
        for c in candidates:
            _, _, j, give, get, _ = c
            if per_player.get(get.id, 0) >= MAX_IDEAS_PER_PLAYER or per_partner.get(j, 0) >= MAX_IDEAS_PER_PARTNER:
                continue
            per_player[get.id] = per_player.get(get.id, 0) + 1
            per_partner[j] = per_partner.get(j, 0) + 1
            shortlist.append(c)
            if len(shortlist) == TRADE_SIM_TOP:
                break
        before = self.playoff_pct(self.dists)
        ideas = []
        for my_gain, their_gain, j, give, get, dists in shortlist:
            cats = self.avg_cat_win(self.idx, dists) - base_cats
            gap = self.model.value(get) - self.model.value(give)
            ideas.append(
                schemas.TradeIdea(
                    partner_team_id=self.teams[j].id,
                    partner_team_name=self.teams[j].name,
                    give=[self.ref(give)],
                    get=[self.ref(get)],
                    my_cats_gain=round(my_gain, 3),
                    their_cats_gain=round(their_gain, 3),
                    paper_gap=round(gap, 2),
                    paper_verdict="win_for_them" if gap <= 0 else "even" if gap <= EVEN_PAPER_GAP else "stretch",
                    playoff_pct_before=round(before, 3),
                    playoff_pct_after=round(self.playoff_pct(dists), 3),
                    category_changes={n: round(float(cats[c]), 3) for c, n in enumerate(self.cat_names)},
                )
            )
        # Small category gains can vanish in the full simulation; only keep moves that hold up there.
        ideas = [t for t in ideas if t.playoff_pct_after >= t.playoff_pct_before]
        ideas.sort(key=lambda t: (t.playoff_pct_after, t.my_cats_gain), reverse=True)
        return ideas

    def pickups(self) -> list[schemas.PickupIdea]:
        base = self.model.strength(self.idx, self.dists)
        base_cats = self.avg_cat_win(self.idx, self.dists)
        droppable = [p for p in self.me.roster if p.id not in self.keep]
        candidates = []
        for fa in self.league.free_agents:
            if not fa.available:
                continue
            best = None
            for drop in droppable:
                roster = [p for p in self.me.roster if p.id != drop.id] + [fa]
                dists = list(self.dists)
                dists[self.idx] = self.model.dist(roster)
                gain = self.model.strength(self.idx, dists) - base
                if best is None or gain > best[0]:
                    best = (gain, drop, dists)
            if best and best[0] > 0.02:
                candidates.append((best[0], fa, best[1], best[2]))

        candidates.sort(key=lambda c: c[0], reverse=True)
        before = self.playoff_pct(self.dists)
        ideas = []
        for gain, fa, drop, dists in candidates[:PICKUP_SIM_TOP]:
            cats = self.avg_cat_win(self.idx, dists) - base_cats
            ideas.append(
                schemas.PickupIdea(
                    add=self.ref(fa),
                    drop=self.ref(drop),
                    availability=fa.status,
                    my_cats_gain=round(gain, 3),
                    playoff_pct_before=round(before, 3),
                    playoff_pct_after=round(self.playoff_pct(dists), 3),
                    category_changes={n: round(float(cats[c]), 3) for c, n in enumerate(self.cat_names)},
                )
            )
        ideas = [p for p in ideas if p.playoff_pct_after >= p.playoff_pct_before]
        ideas.sort(key=lambda p: (p.playoff_pct_after, p.my_cats_gain), reverse=True)
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
                    "Trades assume the partner accepts only if their own expected categories won does not drop.",
                ],
            ),
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            elapsed_ms=0,
        )
        report.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return report

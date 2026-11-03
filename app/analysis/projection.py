"""Your matchup this week, projected to Sunday: who takes each category, and who sits on crowded days.

Each day the best healthy players with a game fill the lineup slots (model.py). On a crowded
day more players have games than there are slots, so the rest sit; the projection counts
only the players who start, for both teams. Stats already banked this week are added to the
days left.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.analysis.model import CategoryModel, Dist
from app.espn.parse import League, Player

# A category whose winner is this close to a coin flip is called close.
CLOSE = 0.1


@dataclass(frozen=True)
class CategoryCall:
    stat: str
    mine: float  # projected Sunday total (percentages as fractions)
    theirs: float
    winner: str  # "me" or "them"
    close: bool  # within CLOSE of a coin flip


@dataclass(frozen=True)
class CrowdedDay:
    date: str  # YYYY-MM-DD
    playing: int  # healthy players with a game that day
    slots: int  # lineup slots
    sit: list[tuple[int, str]]  # (player id, name) of who sits: playing but not in the best lineup


@dataclass(frozen=True)
class Projection:
    opponent_id: int
    categories: list[CategoryCall]
    wins: int
    losses: int
    my_crowded: list[CrowdedDay]
    their_crowded: list[CrowdedDay]


def full_week(model: CategoryModel, rest: Dist, banked: dict[int, float]) -> Dist:
    """The days left plus what is already banked this week, as one weekly total."""
    mu, sd = rest.mu.copy(), rest.sd.copy()
    for k, c in enumerate(model.cats):
        if c.ratio is None:
            mu[k] += banked.get(c.stat_id, 0.0)
            continue
        made, tried = (banked.get(s, 0.0) for s in c.ratio)
        left = float(rest.att[k]) if rest.att is not None else 0.0
        if tried + left > 0:
            mu[k] = (made + mu[k] * left) / (tried + left)
            sd[k] = sd[k] * left / (tried + left)
    return Dist(mu, sd, rest.att)


def crowded_days(model: CategoryModel, roster: list[Player], period: int) -> list[CrowdedDay]:
    """Days left this week when more players have a game than the lineup has slots."""
    days, _ = model.horizon[period]
    order = sorted(roster, key=model.value, reverse=True)
    plays = {p.id: model.plays(p, period) for p in order}
    slots = len(model.seats) or model.n_starters
    out = []
    for i, day in enumerate(days):
        playing = [p for p in order if plays[p.id][i]]
        if len(playing) <= slots:
            continue
        started = set(model._fill(tuple(model._seat_mask(p) for p in playing)))
        sit = [(p.id, p.name) for k, p in enumerate(playing) if k not in started]
        if sit:
            out.append(CrowdedDay(model.cal.date_of(day).isoformat(), len(playing), slots, sit))
    return out


def project_matchup(league: League, team_id: int, model: CategoryModel | None = None) -> Projection | None:
    """Your current matchup projected to its last day. None without the NBA calendar or a matchup."""
    model = model or CategoryModel(league)
    period = league.current_period
    if not model.cal or period not in model.horizon:
        return None
    m = next((x for x in league.schedule if x.period == period and team_id in (x.home, x.away)), None)
    if m is None:
        return None
    home = m.home == team_id
    opp_id = m.away if home else m.home
    teams = {t.id: t for t in league.teams}
    if team_id not in teams or opp_id not in teams:
        return None
    me, them = teams[team_id], teams[opp_id]
    mine = full_week(model, model.dist(me.roster, period), m.home_totals if home else m.away_totals)
    theirs = full_week(model, model.dist(them.roster, period), m.away_totals if home else m.home_totals)
    p = model.cat_win_prob(mine, theirs)
    calls = [
        CategoryCall(
            stat=c.name,
            mine=float(mine.mu[k]),
            theirs=float(theirs.mu[k]),
            winner="me" if p[k] >= 0.5 else "them",
            close=bool(abs(p[k] - 0.5) <= CLOSE),
        )
        for k, c in enumerate(model.cats)
    ]
    wins = int(np.sum(p >= 0.5))
    return Projection(
        opponent_id=opp_id,
        categories=calls,
        wins=wins,
        losses=len(calls) - wins,
        my_crowded=crowded_days(model, me.roster, period),
        their_crowded=crowded_days(model, them.roster, period),
    )

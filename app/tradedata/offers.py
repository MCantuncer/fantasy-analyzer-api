"""Turn a league-season's public transactions into trade offers with known outcomes.

What ESPN shows a stranger about a public league's trades:
  * TRADE_PROPOSAL, status CANCELED: an offer that was withdrawn or expired, with its full
    contents. Nobody accepted it while it was open.
  * TRADE_ACCEPT: someone accepted an offer, but without its contents (the proposal it points
    to is hidden). The contents come from the daily rosters: players who go straight from one
    team to another between two days were traded.
  * TRADE_DECLINE: someone declined an offer, contents hidden. Kept as counts per team only.
  * TRADE_VETO / TRADE_UPHOLD: league votes on an accepted trade.

So the offers with contents are the accepted ones and the withdrawn ones. Declined offers are
the strongest "no" but cannot be used for contents, which the report states as a gap.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

ACCEPTED = "accepted"  # accepted and the players moved
WITHDRAWN = "withdrawn"  # proposer cancelled it, or it expired, without the other side accepting
ACCEPTED_NOT_PROCESSED = "accepted_not_processed"  # accepted, but the players never moved (vetoed or voided)
EXECUTION_WINDOW = 5  # days between an accept and the roster move (review/veto periods)


@dataclass
class Offer:
    league_id: int
    season: int
    offer_id: str
    day: int  # scoring period of the decision: the proposal for withdrawn offers, the accept for trades
    proposed_ms: int  # ESPN timestamp of that event
    proposer: int  # team that made the offer (for accepted trades: the side that did not accept)
    receivers: list[int]  # the other team(s)
    outcome: str
    # (player id, from team, to team) for every player in the offer.
    moves: list[tuple[int, int, int]] = field(default_factory=list)
    drops: list[tuple[int, int]] = field(default_factory=list)  # (player, team) dropped to make room
    source: str = ""  # "proposal" or "rosters"
    vetoes: int = 0
    prior_offers: int = 0  # earlier offers (any outcome, contents known) between the same teams this season
    repeat: bool = False  # same players as one of those: a re-offer, not a fresh one
    notes: list[str] = field(default_factory=list)

    @property
    def teams(self) -> list[int]:
        return [self.proposer, *self.receivers]


def _trade_events(transactions: dict[int, list[dict]]) -> list[dict]:
    seen, out = set(), []
    for day in sorted(transactions):
        for t in transactions[day]:
            if t.get("type", "").startswith("TRADE") and t["id"] not in seen:
                seen.add(t["id"])
                out.append(t)
    return out


def roster_moves(
    rosters: dict[int, dict[int, list[list[int]]]], adds: set[tuple[int, int, int]] = frozenset()
) -> dict[int, list[tuple[int, int, int]]]:
    """Players who changed fantasy teams overnight, by the first day they show on the new team.

    `adds` holds (player, team, day) of free-agent and waiver pickups: a player dropped and picked
    up by another team between two roster days also looks like a direct move, and is no trade.
    """
    days = sorted(rosters)
    out: dict[int, list[tuple[int, int, int]]] = {}
    prev: dict[int, int] | None = None
    prev_day = 0
    for d in days:
        cur = {pid: tid for tid, entries in rosters[d].items() for pid, _slot in entries}
        if prev is not None:
            moved = [
                (pid, prev[pid], tid)
                for pid, tid in cur.items()
                if pid in prev and prev[pid] != tid and not any((pid, tid, x) in adds for x in range(prev_day, d + 1))
            ]
            if moved:
                out[d] = sorted(moved)
        prev, prev_day = cur, d
    return out


def draft_rosters(league: dict) -> dict[int, list[list[int]]]:
    """Each team's players as the draft left them: the day-0 roster, for trades made before opening night."""
    out: dict[int, list[list[int]]] = defaultdict(list)
    for p in (league.get("draftDetail") or {}).get("picks", []):
        if p.get("playerId", -1) > 0 and p.get("teamId"):
            out[int(p["teamId"])].append([int(p["playerId"]), -1])
    return dict(out)


def _adds(events: list[dict]) -> set[tuple[int, int, int]]:
    return {
        (int(i["playerId"]), int(i["toTeamId"]), int(t["scoringPeriodId"]))
        for t in events
        if t.get("type") in ("FREEAGENT", "WAIVER") and t.get("status") == "EXECUTED"
        for i in t.get("items") or []
        if i.get("type") == "ADD"
    }


def _components(moves: list[tuple[int, int, int]]) -> list[list[tuple[int, int, int]]]:
    """Split one day's moves into separate trades: teams linked by a moved player are one trade."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _pid, a, b in moves:
        parent[find(a)] = find(b)
    groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for m in moves:
        groups[find(m[1])].append(m)
    return list(groups.values())


def build_offers(
    league_id: int,
    season: int,
    transactions: dict[int, list[dict]],
    rosters: dict[int, dict[int, list[list[int]]]],
    day0: dict[int, list[list[int]]] | None = None,
    draft_ms: int = 0,
) -> tuple[list[Offer], dict]:
    """Offers with contents, plus counts of what could not be used (declines and unmatched events).

    `day0` is the roster before the first scoring day (the draft), so preseason trades show as moves.
    """
    events = _trade_events(transactions)
    if day0 and 0 not in rosters:
        rosters = {0: day0, **rosters}
    ambiguous: set[str] = set()
    adds = _adds([t for day in transactions.values() for t in day])
    offers: list[Offer] = []
    stats = {
        "declines": 0,
        "declines_by_team": defaultdict(int),
        "unmatched_accepts": 0,
        "duplicate_accepts": 0,
        "pre_draft_accepts": 0,
        "trades_without_accept": 0,
    }

    for t in events:
        if t["type"] == "TRADE_PROPOSAL" and t.get("status") == "CANCELED":
            items = [i for i in t.get("items") or [] if i.get("type") == "TRADE"]
            if not items:
                continue
            proposer = int(t["teamId"])
            others = sorted(({int(i["toTeamId"]) for i in items} | {int(i["fromTeamId"]) for i in items}) - {proposer})
            offers.append(
                Offer(
                    league_id,
                    season,
                    t["id"],
                    int(t["scoringPeriodId"]),
                    int(t.get("proposedDate") or 0),
                    proposer,
                    others,
                    WITHDRAWN,
                    moves=[(int(i["playerId"]), int(i["fromTeamId"]), int(i["toTeamId"])) for i in items],
                    drops=[(int(i["playerId"]), int(i["fromTeamId"])) for i in t["items"] if i.get("type") == "DROP"],
                    source="proposal",
                )
            )
        elif t["type"] == "TRADE_DECLINE":
            stats["declines"] += 1
            stats["declines_by_team"][int(t["teamId"])] += 1

    # Accepted trades: pair each roster move with the latest accept before it by one of its teams
    # (the latest, because an accept that was vetoed can sit just before the one that went through).
    trades = [(d, comp) for d, moves in sorted(roster_moves(rosters, adds).items()) for comp in _components(moves)]
    vetoes: dict[str, int] = defaultdict(int)
    for t in events:
        if t["type"] == "TRADE_VETO":
            vetoes[t.get("relatedTransactionId", "")] += 1
    accepts = []
    for t in events:
        if t["type"] != "TRADE_ACCEPT":
            continue
        if draft_ms and int(t.get("proposedDate") or 0) < draft_ms:
            stats["pre_draft_accepts"] += 1  # off-season keeper or pick deals: the rosters can't show them
            continue
        accepts.append(t)
    used: dict[str, tuple[int, list[tuple[int, int, int]]]] = {}
    matched = 0
    for d, comp in trades:
        teams = {a for _, a, _ in comp} | {b for _, _, b in comp}
        cands = [
            t
            for t in accepts
            if t["id"] not in used
            and int(t["teamId"]) in teams
            and d - EXECUTION_WINDOW <= int(t["scoringPeriodId"]) <= d
        ]
        if cands:
            best = max(cands, key=lambda t: (int(t["scoringPeriodId"]), int(t.get("proposedDate") or 0)))
            used[best["id"]] = (d, comp)
            matched += 1
    for t in accepts:
        day, acceptor = int(t["scoringPeriodId"]), int(t["teamId"])
        drops = [(int(i["playerId"]), int(i["fromTeamId"])) for i in t.get("items") or [] if i.get("type") == "DROP"]
        n_veto = vetoes.get(t.get("relatedTransactionId", ""), 0)
        if t["id"] in used:
            d, comp = used[t["id"]]
            teams = sorted({a for _, a, _ in comp} | {b for _, _, b in comp})
            proposer = next((x for x in teams if x != acceptor), -1)
            offers.append(
                Offer(
                    league_id,
                    season,
                    t["id"],
                    day,
                    int(t.get("proposedDate") or 0),
                    proposer,
                    [x for x in teams if x != proposer],
                    ACCEPTED,
                    moves=comp,
                    drops=drops,
                    source="rosters",
                    vetoes=n_veto,
                    notes=[f"executed day {d}"] + (["multi-team"] if len(teams) > 2 else []),
                )
            )
            continue
        twin = next(
            (
                t2
                for t2 in accepts
                if t2["id"] in used
                and acceptor in {a for _, a, _ in used[t2["id"]][1]} | {b for _, _, b in used[t2["id"]][1]}
                and abs(day - int(t2["scoringPeriodId"])) <= 1
                and n_veto == 0
            ),
            None,
        )
        if twin is not None:  # both sides show an accept for one trade (a countered offer)
            stats["duplicate_accepts"] += 1
            ambiguous.add(twin["id"])
            continue
        stats["unmatched_accepts"] += 1
        offers.append(
            Offer(
                league_id,
                season,
                t["id"],
                day,
                int(t.get("proposedDate") or 0),
                -1,
                [acceptor],
                ACCEPTED_NOT_PROCESSED,
                drops=drops,
                source="accept",
                vetoes=n_veto,
                notes=["players never moved: vetoed, voided, or after the last roster day"],
            )
        )
    for o in offers:
        if o.offer_id in ambiguous:
            o.notes.append("proposer ambiguous: both sides accepted")
    stats["trades_without_accept"] = len(trades) - matched
    stats["declines_by_team"] = dict(stats["declines_by_team"])
    _mark_repeats(offers)
    return offers, stats


def _mark_repeats(offers: list[Offer]) -> None:
    """Flag offers that repeat an earlier one between the same teams: a negotiation, not a fresh offer."""
    history: dict[frozenset[int], list[Offer]] = defaultdict(list)
    for o in sorted(offers, key=lambda o: o.proposed_ms):
        key = frozenset(o.teams)
        same = [p for p in history[key] if set(p.moves) == set(o.moves) and o.moves]
        o.repeat = bool(same)
        o.prior_offers = len(history[key])
        history[key].append(o)

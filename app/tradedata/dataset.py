"""Cached league-seasons -> offer rows (JSON lines) plus a flat CSV and a per-season summary."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from app.backtest.history import History
from app.tradedata.features import CATS, Snapshot, offer_row
from app.tradedata.fetch import load_json
from app.tradedata.offers import build_offers


def build_league_season(league_id: int, season: int, cache: Path) -> tuple[list[dict], dict]:
    tx = {int(k): v for k, v in load_json(cache / "tx" / f"{league_id}-{season}.json.gz").items()}
    h = History.load(cache / "history" / f"{league_id}-{season}.json.gz")
    meta_path = cache / "meta" / f"{league_id}-{season}.json.gz"
    meta = {int(k): v for k, v in load_json(meta_path).items()} if meta_path.exists() else {}
    offers, stats = build_offers(league_id, season, tx, h.rosters)
    snap = Snapshot(h, meta)
    events = sorted((t for day in tx.values() for t in day), key=lambda t: t.get("proposedDate") or 0)
    rows = []
    for o in sorted(offers, key=lambda o: o.proposed_ms):
        before = [e for e in events if (e.get("proposedDate") or 0) < o.proposed_ms]
        rows.append(offer_row(snap, o, before))
    settings = h.league["settings"]
    stats |= {
        "league_id": league_id,
        "season": season,
        "name": settings.get("name", ""),
        "scoring_type": settings["scoringSettings"].get("scoringType"),
        "teams": len(h.league.get("teams", [])),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "has_prior_season": bool(h.prior),
        "has_projection": any(m.get("projection") for m in meta.values()),
        "box_days": len(h.box),
        "roster_days": len(h.rosters),
    }
    return rows, stats


def flat(row: dict) -> list[dict]:
    """One CSV line per (offer, side): the side's view of the deal, for quick looks in a spreadsheet."""
    out = []
    teams = {t["team_id"]: t for t in row["teams"]}
    for tid, side in row["sides"].items():
        t = teams.get(int(tid), {})
        role = "proposer" if int(tid) == row["proposer"] else "receiver"
        out.append(
            {
                "league_id": row["league_id"],
                "season": row["season"],
                "offer_id": row["offer_id"],
                "day": row["day"],
                "season_fraction": row["season_fraction"],
                "outcome": row["outcome"],
                "scoring_type": row["league"]["scoring_type"],
                "team_id": tid,
                "role": role,
                "n_give": len(side["gives"]),
                "n_get": len(side["gets"]),
                "net_season": side["net_season"],
                "net_last7": side["net_last7"],
                "net_last15": side["net_last15"],
                "net_prior_season": side["net_prior_season"],
                "net_projection_value": side["net_projection_value"],
                "best_given_rank": side["best_player_given_rank"],
                "best_got_rank": side["best_player_got_rank"],
                "win_pct": t.get("win_pct"),
                "standing_rank": t.get("rank"),
                "playoff_line_gap": t.get("playoff_line_gap"),
                "roster_value": t.get("roster_value"),
                **{f"strength_{c}": (t.get("category_strength") or {}).get(c) for c in CATS},
                "prior_offers_same_teams": row["prior_offers_same_teams"],
                "repeat": row["repeat"],
            }
        )
    return out


def write(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "offers.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    lines = [x for r in rows for x in flat(r)]
    if lines:
        with (out_dir / "offer_sides.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(lines[0]))
            w.writeheader()
            w.writerows(lines)

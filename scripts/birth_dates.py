"""Rebuilds app/espn/birth_dates.csv, the players' birth dates behind their ages.

    uv run python -m scripts.birth_dates

ESPN's fantasy API has no ages, so they come from Basketball-Reference's player register,
as published in the sumitrodatta/bball-reference-datasets repository on GitHub (updated
during the season, rookies included). Only players who played in 2019-20 or later are kept,
the seasons ESPN's API covers. Names are matched to ESPN's by app/espn/ages.name_key; when
two kept players share a key, the one who played most recently wins. Run it once a season,
after the draft, so rookies get an age.
"""

from __future__ import annotations

import csv
import io

import httpx

from app.espn.ages import DATA, name_key

SOURCE = (
    "https://raw.githubusercontent.com/sumitrodatta/bball-reference-datasets/master/Data/Player%20Career%20Info.csv"
)
FIRST_SEASON = 2020  # bbref labels a season by the year it ends, as ESPN's seasonId does


def main() -> None:
    resp = httpx.get(SOURCE, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    kept: dict[str, tuple[int, str, str]] = {}
    for r in csv.DictReader(io.StringIO(resp.text)):
        born, last = r.get("birth_date") or "", r.get("to") or ""
        if not born or born == "NA" or not last.isdigit() or int(last) < FIRST_SEASON:
            continue
        key = name_key(r["player"])
        if key not in kept or int(last) > kept[key][0]:
            kept[key] = (int(last), r["player"], born)
    rows = sorted((name, born) for _, name, born in kept.values())
    with DATA.open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["name", "birth_date"])
        out.writerows(rows)
    print(f"{len(rows)} players written to {DATA}")


if __name__ == "__main__":
    main()

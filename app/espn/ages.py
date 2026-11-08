"""Players' ages, which ESPN's fantasy API doesn't give.

Birth dates live in birth_dates.csv next to this file (rebuilt by scripts/birth_dates.py)
and are looked up by name, since that's what ESPN and the source share. A player who isn't
in the file has no age, and nothing that uses ages changes for him.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from datetime import date
from functools import cache
from pathlib import Path

DATA = Path(__file__).with_name("birth_dates.csv")
_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
# ESPN names that Basketball-Reference spells another way (keys on both sides).
ALIASES = {"ronald holland": "ron holland"}


def name_key(name: str) -> str:
    """A name with accents, punctuation and suffixes dropped: "Nikola Jokić" and "Nikola Jokic",
    "Jaren Jackson Jr." and "Jaren Jackson" match."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    plain = " ".join(_SUFFIX.sub("", re.sub(r"[.'’\-]", "", plain)).split())
    return ALIASES.get(plain, plain)


@cache
def _births() -> dict[str, date]:
    if not DATA.exists():
        return {}
    with DATA.open(newline="") as f:
        return {name_key(r["name"]): date.fromisoformat(r["birth_date"]) for r in csv.DictReader(f)}


def birth_date(name: str) -> date | None:
    return _births().get(name_key(name))


def season_age(name: str, season: int) -> float | None:
    """Age in years at the season's midpoint, February 1 of the year ``season`` ends in
    (ESPN's seasonId is that year), so it stays the same all season."""
    born = birth_date(name)
    return None if born is None else (date(season, 2, 1) - born).days / 365.25

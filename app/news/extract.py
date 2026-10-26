"""Who a news item is about and what kind of news it is.

Precision beats recall here: a player is tagged only when his full name appears
(or ESPN tags him), never from a last name alone, and a name two players share is
skipped. The kind and the injury severity come from plain keyword rules, so every
tag can be explained by the words in the headline.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

Kind = Literal["injury", "return", "trade", "rumor", "signing", "waive", "role", "other"]

# A headline naming more players than this is a roundup (rankings, recaps), not news about each of them.
MAX_PLAYERS = 3
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize(text: str) -> str:
    """Lowercase ASCII words: accents, dots, apostrophes and suffixes like "Jr." removed."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[.'’`]", "", text.lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _name_key(full_name: str) -> str:
    words = [w for w in normalize(full_name).split() if w not in SUFFIXES]
    return " ".join(words)


class NameIndex:
    """Finds players by full name in normalized text."""

    def __init__(self, names: dict[int, str]):
        seen: dict[str, int | None] = {}
        for pid, name in names.items():
            key = _name_key(name)
            if len(key.split()) < 2:
                continue
            seen[key] = None if key in seen and seen[key] != pid else pid
        self._ids = {k: v for k, v in seen.items() if v is not None}
        keys = sorted(self._ids, key=len, reverse=True)
        self._re = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b") if keys else None

    def find(self, text: str) -> list[int]:
        if self._re is None:
            return []
        words = " ".join(w for w in normalize(text).split() if w not in SUFFIXES)
        out: list[int] = []
        for m in self._re.finditer(words):
            pid = self._ids[m.group(1)]
            if pid not in out:
                out.append(pid)
        return out


def _any(pattern: str) -> re.Pattern:
    return re.compile(pattern)


RUMOR = _any(
    r"\b(rumou?rs?|interest(ed)?|talks|discuss(es|ed|ing|ions)?|monitor(ing)?|explor(e|es|ing)|"
    r"eyeing|eye|pursu(e|ing|it)|target(s|ing)?|linked|could|might|would|open to|shopping|shop|"
    r"available|trade market|trade block|frontrunners?|suitors?)\b"
)
TRADE = _any(r"\b(trade[sd]?|trading|acquire[sd]?|acquiring|deal sends?|swap)\b")
SIGNING = _any(r"\b(signs?|signed|signing|agrees?|agreed|re signs?|two way|10 day|contract|extension)\b")
WAIVE = _any(r"\b(waive[sd]?|waiving|release[sd]?|releasing|cut)\b")
RETURN = _any(
    r"\b(returns?|returned|returning|cleared|upgraded|activated|back in (the )?lineup|will play|"
    r"set to play|expected to play|available to play|makes? (his )?(season )?debut)\b"
)
ROLE = _any(
    r"\b(starting lineup|start(s|ed|ing)? (for|in place|at)|first start|moves? to the bench|"
    r"to the bench|benched|rotation|minutes restriction|minutes limit|bigger role|larger role|promoted)\b"
)
INJURY = _any(
    r"\b(injur(y|ies|ed)|ruled out|out (for|indefinitely|tonight|again)|will miss|miss(es|ed)?|sidelined|"
    r"sprain(ed)?|strain(ed)?|fracture[sd]?|broken|torn|tear|ruptured?|surgery|soreness|sore|tightness|"
    r"contusion|bruise[sd]?|concussion|mri|x rays?|questionable|doubtful|probable|day to day|"
    r"week to week|game time decision|illness|dislocat\w+|separated|hyperextend\w*)\b"
)

SEVERE = _any(
    r"\b(fracture[sd]?|broken|break|torn|tear|ruptured?|surgery|season ending|indefinitely|"
    r"out for (the )?season|\d+ (to \d+ )?(weeks|months)|multiple weeks|several weeks|"
    r"weeks|months|separated|dislocat\w+)\b"
)
MODERATE = _any(r"\b(ruled out|out|will miss|sprain(ed)?|strain(ed)?|week to week|doubtful|concussion|sidelined)\b")
MILD = _any(
    r"\b(questionable|day to day|probable|soreness|sore|tightness|bruise[sd]?|contusion|game time decision|rest)\b"
)
BODY_PARTS = (
    "ankle knee wrist hand finger thumb foot toe heel achilles hamstring calf quad quadriceps groin hip back "
    "shoulder elbow neck rib ribs abdominal oblique adductor hamstring shin eye nose face illness concussion"
).split()
_BODY = re.compile(r"\b(" + "|".join(BODY_PARTS) + r")\b")

SEVERITY_LABEL = {0: "no injury", 1: "minor, day-to-day", 2: "out, short term", 3: "serious, weeks or more"}


def kind_of(text: str) -> Kind:
    t = normalize(text)
    move = TRADE.search(t) or SIGNING.search(t)
    if move and RUMOR.search(t):
        return "rumor"
    if TRADE.search(t):
        return "trade"
    if WAIVE.search(t):
        return "waive"
    if SIGNING.search(t):
        return "signing"
    if INJURY.search(t) and (SEVERE.search(t) or MODERATE.search(t) or not RETURN.search(t)):
        return "injury"
    if RETURN.search(t):
        return "return"
    if ROLE.search(t):
        return "role"
    return "other"


def severity(text: str) -> int:
    """0 none, 1 minor (day-to-day), 2 out short term, 3 serious (weeks or more, surgery, fracture)."""
    t = normalize(text)
    if SEVERE.search(t):
        return 3
    if MODERATE.search(t):
        return 2
    if MILD.search(t):
        return 1
    return 0


def body_part(text: str) -> str | None:
    m = _BODY.search(normalize(text))
    return m.group(1) if m else None


@dataclass
class Tags:
    kind: Kind
    player_ids: list[int]  # the players the item is about, at most MAX_PLAYERS
    severity: int
    body_part: str | None


def tag(title: str, summary: str, index: NameIndex, espn_ids: list[int] | None = None) -> Tags:
    """Players come from the headline (or ESPN's own tags); the kind and severity read the summary too."""
    players = list(espn_ids or []) or index.find(title)
    if not players:
        players = index.find(summary)[:1]  # the summary's first name, only when the headline names nobody
    if len(players) > MAX_PLAYERS:
        players = []
    text = f"{title}. {summary}"
    kind = kind_of(title)
    if kind == "other":
        kind = kind_of(text)
    sev = severity(text) if kind == "injury" else 0
    return Tags(kind=kind, player_ids=players, severity=sev, body_part=body_part(text) if kind == "injury" else None)

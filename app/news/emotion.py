"""Emotion words in news text, kept for a test still to come.

Can 2026-11-19: does what is going on around a player (a benching, a feud, a trade demand, a family
matter, a milestone) move his next games? Five seasons of box scores say such things show up through
his minutes when they show up at all, but news text itself could not be tested: the news table keeps
14 days. So every outlet item that names a player is also written to a log that is never trimmed
(table ``news_log``), with the tags below. In a few weeks the tagged players' next games can be
checked against their lines, blind.

The tags are plain keyword rules on the same normalized text as ``extract.py``, so each one can be
explained by the words in the item. They are a first cut (about 6 in 7 right on 227 real headlines):
the log keeps the title and summary so the text can be read again with better rules, or by a model.
Nothing here reaches the app's pages or its numbers.
"""

from __future__ import annotations

import re

from app.news.extract import normalize

TAGS: dict[str, re.Pattern] = {
    # League or team discipline, temper on the court.
    "discipline": re.compile(
        r"\b(suspen(d|ds|ded|sion)|fined|fine of|ejected|ejection|flagrant|technical fouls?|altercation|"
        r"scuffle|fight|shov(e|ed|ing)|elbow(ed|ing)|punch(ed|es)?|conduct detrimental)\b"
    ),
    # Out of the coach's plans.
    "benched": re.compile(
        r"\b(benched|benching|demoted|moves? to (the )?bench|out of (the )?rotation|"
        r"removed from (the )?(starting lineup|rotation)|healthy scratch|dnp cd|coachs decision|lack of effort|"
        r"barely seen the floor|lose[sd]? (his )?starting (job|spot))\b"
    ),
    # Wants out, or trade talk around him.
    "trade_unrest": re.compile(
        r"\b(wants out|want out|trade request|requests? (a )?trade|demand(s|ed)? (a )?trade|demanded \w+ trade|"
        r"unhappy|not happy|disgruntled|trade block|on the block|trade rumou?rs?|exploring trades?|"
        r"contemplating (his )?future|shopping)\b"
    ),
    "contract": re.compile(
        r"\b(extension|contract year|player option|team option|declines? (his )?option|decline option|opts? out|"
        r"opt(s|ed)? in|max (deal|contract)|supermax|holdout|contract (talks|dispute)|free agency|expiring)\b"
    ),
    # Away for personal reasons, family.
    "personal": re.compile(
        r"\b(personal( reasons)?|family (matter|reasons|emergency)|birth of|newborn|paternity|bereavement|"
        r"funeral|death of|passing of|mourning|away from (the )?team|not with (the )?team|leave of absence)\b"
    ),
    # Anger, frustration, public friction with a coach, a teammate, an opponent or the fans.
    "friction": re.compile(
        r"\b(frustrat\w+|bristles?|calls? out|called out|rips?|ripped|blasts?|blasted|criticiz\w+|unacceptable|"
        r"lost all respect|sucks|booed|boos|trash talk\w*|beef|drama|feud|not welcome|angry|upset with|fed up|"
        r"pushes back|false allegations|heated|confront\w+|be better|implores|everything falls on)\b"
    ),
    # Pride and statement games.
    "motivation": re.compile(
        r"\b(statement|revenge|motivat\w+|chip on (his )?shoulder|prove|redemption|homecoming|former team|"
        r"old team|emotional|tears|dedicates?|welcome to|still hurt)\b"
    ),
    # Milestones, record chases, honours.
    "milestone": re.compile(
        r"\b(milestone|record books?|franchise record|all time|career high|(scoring|point|double double) streak|"
        r"retire[sd]? .*(jersey|no \d+|number)|jersey retire\w*|number retirement|"
        r"only \d+ players? in nba history|passed \w+ \w+)\b"
    ),
    # Trouble off the court.
    "legal": re.compile(
        r"\b(arrest(ed)?|charged|plea|indict(ed|ment)?|lawsuit|investigation|betting case|domestic|dui)\b"
    ),
}


def tags(title: str, summary: str = "") -> list[str]:
    """The emotion tags whose words appear in the item's title or summary, in TAGS order."""
    text = normalize(f"{title}. {summary}")
    return [name for name, pattern in TAGS.items() if pattern.search(text)]

from datetime import date

import pytest

from app.espn.ages import birth_date, name_key, season_age


@pytest.mark.parametrize(
    ("espn", "bbref"),
    [
        ("Nikola Jokić", "Nikola Jokic"),
        ("Jaren Jackson Jr.", "Jaren Jackson"),
        ("Kelly Oubre Jr.", "Kelly Oubre"),
        ("Shai Gilgeous-Alexander", "Shai GilgeousAlexander"),
        ("De'Aaron Fox", "DeAaron Fox"),
        ("Ronald Holland II", "Ron Holland"),  # an alias
    ],
)
def test_names_match_across_spellings(espn, bbref):
    assert name_key(espn) == name_key(bbref)


def test_birth_dates_and_season_age():
    assert birth_date("Stephon Castle") == date(2004, 11, 1)
    assert birth_date("Nikola Jokić") is not None
    assert season_age("Stephon Castle", 2027) == pytest.approx(22.25, abs=0.01)
    assert season_age("Stephon Castle", 2026) == pytest.approx(21.25, abs=0.01)
    assert season_age("Not A Player", 2027) is None

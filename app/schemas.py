"""API request and response models. The client generates its types from these."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LeagueRequest(BaseModel):
    league_id: int = Field(gt=0)
    season: int | None = Field(default=None, description="ESPN season year, e.g. 2027 for 2026-27")


class AnalyzeRequest(LeagueRequest):
    team_id: int = Field(gt=0)
    keep_player_ids: list[int] = Field(
        default_factory=list, description="Players never offered in trades or suggested as drops"
    )


class TeamSummary(BaseModel):
    id: int
    name: str
    abbrev: str
    record: str


class LeagueSummary(BaseModel):
    id: int
    season: int
    name: str
    scoring_type: str
    categories: list[str]
    teams: list[TeamSummary]
    playoff_teams: int
    is_public: bool
    supported: bool
    unsupported_reason: str | None = None


class PlayerRef(BaseModel):
    id: int
    name: str
    positions: list[str]
    injury_status: str
    percent_owned: float
    value: float = Field(description="Generic z-score value summed over the league's categories")
    perceived_value: float = Field(
        description="Trade value in auction dollars: draft-day signals fading into % owned, "
        "rest-of-season category value and this season's production"
    )
    tier: Literal["core", "tradeable", "droppable", "injured"] = Field(
        description="core: kept by you. tradeable: has a market, trade rather than drop. "
        "droppable: replaceable from free agency. injured: out right now."
    )


class CategoryOutlook(BaseModel):
    category: str
    win_pct: float = Field(description="Chance of winning this category in an average week")
    league_rank: int = Field(description="1 = best weekly projection in the league")
    weekly_projection: float
    tag: Literal["locked", "swing", "lost"]


class TeamBuild(BaseModel):
    team_id: int
    team_name: str
    ranks: dict[str, int]
    strong: list[str]
    weak: list[str] = Field(description="Categories a team is far behind in, likely punts")
    expected_cats_won: float


class TradeIdea(BaseModel):
    partner_team_id: int
    partner_team_name: str
    give: list[PlayerRef]
    get: list[PlayerRef]
    my_drop: PlayerRef | None = Field(default=None, description="Who you cut to make room when you get more players")
    their_drop: PlayerRef | None = Field(default=None, description="Who they would likely cut to make room")
    my_cats_gain: float = Field(description="Change in my expected categories won per week")
    their_cats_gain: float = Field(description="Change in theirs, ignoring the categories they punt")
    value_they_get: float = Field(description="Perceived value they receive, in auction dollars")
    value_they_give: float = Field(description="Perceived value they send, in auction dollars")
    acceptance: float = Field(description="Rough chance the other owner accepts (heuristic)")
    acceptance_label: Literal["likely", "possible", "long_shot"]
    pitch: str = Field(description="Why the deal helps them, to use when you propose it")
    reasons: list[str] = Field(description="Why it can work for both sides: sell high, buy low, their punts, timing")
    playoff_pct_before: float
    playoff_pct_after: float
    category_changes: dict[str, float] = Field(description="Change in my weekly win % per category")


class PickupIdea(BaseModel):
    add: PlayerRef
    drop: PlayerRef
    open_spot_by_trade: bool = Field(
        description="The spot would come from trading `drop` (he has trade value), not from releasing him"
    )
    availability: str
    my_cats_gain: float
    playoff_pct_before: float
    playoff_pct_after: float
    category_changes: dict[str, float]


class SeasonRow(BaseModel):
    team_id: int
    team_name: str
    current_record: str
    projected_wins: float
    projected_losses: float
    projected_ties: float
    playoff_pct: float
    first_seed_pct: float
    mean_rank: float


class Assumptions(BaseModel):
    games_per_week: float
    simulations: int
    scoring_type: str
    notes: list[str]


class AnalysisReport(BaseModel):
    league: LeagueSummary
    team_id: int
    team_name: str
    roster: list[PlayerRef]
    categories: list[CategoryOutlook]
    builds: list[TeamBuild]
    trades: list[TradeIdea]
    pickups: list[PickupIdea]
    season: list[SeasonRow]
    faab_remaining: int | None
    assumptions: Assumptions
    generated_at: str
    elapsed_ms: int


class ErrorBody(BaseModel):
    code: str
    message: str

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
    not_for_sale_player_ids: list[int] = Field(
        default_factory=list, description="Other teams' players their owners won't move: never targeted in trades"
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


class PlayerBadge(BaseModel):
    kind: Literal["jewel", "keystone", "fresh_signing", "not_for_sale", "on_the_block"] = Field(
        description="jewel: star name. keystone: carries a category for his team. fresh_signing: just traded for "
        "or picked up. not_for_sale: untouchable. on_the_block: on his owner's ESPN trade block."
    )
    category: str | None = Field(default=None, description="The category a keystone carries")
    reason: str


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
    badges: list[PlayerBadge] = Field(default_factory=list, description="Why his owner holds on to him")
    owner_premium: float = Field(
        default=0.0, description="What his owner wants on top of trade value, in auction dollars"
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
    my_add: PlayerRef | None = Field(
        default=None, description="Free agent you add to fill the spot a 2-for-1 opens on your roster"
    )
    my_cats_gain: float = Field(description="Change in my expected categories won per week")
    their_cats_gain: float = Field(description="Change in theirs, ignoring the categories they punt")
    edge: float = Field(description="My gain minus theirs: how much more the deal does for me")
    value_they_get: float = Field(description="Perceived value they receive, in auction dollars")
    value_they_give: float = Field(
        description="Perceived value they send, in auction dollars, including their owner premium"
    )
    owner_premium: float = Field(
        default=0.0, description="Part of value_they_give that is owner premium on their badged players"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Jewels or keystones you give up: only suggested for a clearly better deal"
    )
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


class TradeTarget(BaseModel):
    player: PlayerRef
    my_cats_gain: float = Field(description="Change in my expected categories won if I added him")


class TeamRoster(BaseModel):
    team_id: int
    team_name: str
    players: list[TradeTarget]


class AnalysisReport(BaseModel):
    league: LeagueSummary
    team_id: int
    team_name: str
    roster: list[PlayerRef]
    categories: list[CategoryOutlook]
    builds: list[TeamBuild]
    trades: list[TradeIdea] = Field(description="Fair deals: both teams' categories improve")
    edge_trades: list[TradeIdea] = Field(
        description="Edge deals: they get at least equal trade value on paper, you gain clearly more on the court"
    )
    pickups: list[PickupIdea]
    season: list[SeasonRow]
    league_rosters: list[TeamRoster] = Field(description="Every other team's players, for picking a trade target")
    faab_remaining: int | None
    assumptions: Assumptions
    generated_at: str
    elapsed_ms: int


class ErrorBody(BaseModel):
    code: str
    message: str


class TradeSide(BaseModel):
    team_id: int
    team_name: str
    abbrev: str
    sends: list[PlayerRef]
    drop: PlayerRef | None = Field(default=None, description="Who this team cuts to make room")
    cats_gain: float = Field(description="Change in expected categories won per week, ignoring this team's punts")
    category_changes: dict[str, float] = Field(description="Change in weekly win % per category")
    punts: list[str]
    acceptance: float = Field(description="Rough chance this owner says yes (heuristic)")
    acceptance_label: Literal["likely", "possible", "long_shot"]
    pitch: str = Field(description="Why the deal helps this team")
    reasons: list[str]
    playoff_pct_before: float
    playoff_pct_after: float


class PollTally(BaseModel):
    yes: int = 0
    no: int = 0


class TradeOfTheDay(BaseModel):
    status: Literal["ready", "pending", "none"] = Field(
        description="pending: still being worked out, ask again shortly; none: no deal works for both sides today"
    )
    date: str = Field(description="UTC day this pick is for, YYYY-MM-DD")
    league_id: int
    season: int
    headline: str | None = None
    sides: list[TradeSide] = Field(default_factory=list, description="The two teams; the spotlight team first")
    spotlight_team_id: int | None = None
    team_pairs: int = 0
    win_win_deals: int = Field(default=0, description="Deals found that work for both sides")
    votes: dict[int, PollTally] = Field(default_factory=dict, description="'Would you take it?' votes, by team id")
    generated_at: str | None = None


class VoteRequest(LeagueRequest):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    team_id: int = Field(gt=0, description="The side being voted on")
    take: bool = Field(description="True: that team should take the deal")
    voter: str = Field(
        min_length=8, max_length=64, pattern=r"^[A-Za-z0-9-]+$", description="Random id kept by the browser"
    )


class TeamLinkStart(LeagueRequest):
    team_id: int = Field(gt=0)


class TeamLinkChallenge(BaseModel):
    challenge: str = Field(description="Signed challenge to send back to /api/team-link/verify")
    code: str = Field(description="Set this as the team's abbreviation on ESPN")
    expires_at: str
    team_id: int
    team_name: str
    current_abbrev: str


class TeamLinkVerify(BaseModel):
    challenge: str = Field(min_length=10, max_length=512)


class LinkedTeam(BaseModel):
    league_id: int
    season: int
    team_id: int
    team_name: str
    token: str | None = Field(
        default=None, description="Only on a fresh link: send it as 'Authorization: Bearer <token>' from now on"
    )


class FeedPlayer(BaseModel):
    id: int
    name: str
    position: str


class FeedMove(BaseModel):
    player: FeedPlayer
    kind: Literal["add", "drop", "trade"]
    from_team_id: int = Field(description="0 = free agency")
    to_team_id: int = Field(description="0 = free agency")


class FeedBid(BaseModel):
    team_id: int
    bid: int


class Reactions(BaseModel):
    counts: dict[str, int] = Field(default_factory=dict)
    mine: list[str] = Field(default_factory=list, description="Reactions from the linked team")


class FeedItem(BaseModel):
    id: str = Field(description="tx:<id> for an ESPN move, post:<id> for a message; also the reply/reaction target")
    kind: Literal["waiver", "free_agent", "trade", "post"]
    at: str = Field(description="ISO time")
    team_id: int = Field(description="The team that made the move or wrote the post")
    team_ids: list[int] = Field(default_factory=list, description="Every team in the move")
    bid: int | None = None
    moves: list[FeedMove] = Field(default_factory=list)
    losing_bids: list[FeedBid] = Field(default_factory=list, description="Bids this claim beat, highest first")
    body: str | None = None
    reactions: Reactions = Field(default_factory=Reactions)
    replies: int = 0


class FeedTeam(BaseModel):
    id: int
    name: str
    abbrev: str
    faab_spent: int | None = None


class FeedRequest(LeagueRequest):
    page: int = Field(default=0, ge=0, le=60, description="0 = the last 7 days, 1 = the 7 before that, ...")


class FeedPage(BaseModel):
    league_id: int
    season: int
    page: int
    has_more: bool
    items: list[FeedItem]
    teams: list[FeedTeam]
    faab_budget: int | None = None
    me: LinkedTeam | None = Field(default=None, description="The linked team; chat and reactions need one")


class RepliesRequest(LeagueRequest):
    target: str = Field(pattern=r"^(tx|post):[A-Za-z0-9-]{1,64}$")


class RepliesPage(BaseModel):
    target: str
    items: list[FeedItem]


class PostRequest(LeagueRequest):
    body: str = Field(min_length=1, max_length=500)
    target: str | None = Field(default=None, pattern=r"^(tx|post):[A-Za-z0-9-]{1,64}$", description="None = chat")


class DeletePostRequest(LeagueRequest):
    post_id: int = Field(gt=0)


class ReactRequest(LeagueRequest):
    target: str = Field(pattern=r"^(tx|post):[A-Za-z0-9-]{1,64}$")
    reaction: str = Field(pattern=r"^(steal|smart|overpay|lol|won:\d{1,4})$")
    on: bool = True


class ReactionsOut(BaseModel):
    target: str
    reactions: Reactions


class TargetRequest(AnalyzeRequest):
    player_id: int = Field(gt=0, description="The rostered player you want to trade for")


class TargetFit(BaseModel):
    label: Literal["great", "good", "small", "none"]
    my_cats_gain: float = Field(description="Change in my expected categories won per week if I add him")
    my_drop: PlayerRef | None = Field(description="Who you'd cut to fit him if you gave nothing up")
    helps: list[str] = Field(description="Categories he improves most for you, best first")
    hurts: list[str] = Field(description="Categories he makes worse for you")
    playoff_pct_before: float
    playoff_pct_after: float = Field(description="Playoff odds if you added him for free, as a ceiling")
    category_changes: dict[str, float] = Field(description="Change in my weekly win % per category")


class OwnerWant(BaseModel):
    player: PlayerRef
    their_cats_gain: float = Field(description="Change in their expected categories won, ignoring their punts")
    helps: list[str]


class OwnerRead(BaseModel):
    team_id: int
    team_name: str
    record: str
    build: TeamBuild
    punts: list[str] = Field(description="Categories they win less than 30% of the time")
    needs: list[str] = Field(description="Close categories they compete in, where a gain flips weeks")
    playoff_pct: float
    outside_playoffs: bool
    target_is_star: bool = Field(description="The target is one of the league's jewels: his owner wants a premium")
    would_value: list[OwnerWant] = Field(description="Your players that help their build most")


class TargetOffer(TradeIdea):
    angle: Literal["best", "safer", "cheaper", "alternative"]
    angle_note: str


class TargetReport(BaseModel):
    team_id: int
    team_name: str
    target: PlayerRef
    fit: TargetFit
    owner: OwnerRead
    offers: list[TargetOffer] = Field(description="Best offer first, then up to two fallbacks")
    note: str | None = Field(default=None, description="Why there are no offers, when there are none")
    generated_at: str
    elapsed_ms: int

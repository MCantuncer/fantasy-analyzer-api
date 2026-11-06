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
    draft_type: Literal["auction", "snake"] = "auction"
    round_values: list[float] | None = Field(
        default=None,
        description="Snake leagues: trade value of the last pick in each draft round, round 1 first. "
        "A trade value at or above entry k is worth a round k+1 pick. Null in auction leagues.",
    )


class PlayerBadge(BaseModel):
    kind: Literal["jewel", "keystone", "fresh_signing", "hot", "not_for_sale", "on_the_block"] = Field(
        description="jewel: star name. keystone: carries a category for his team. fresh_signing: just traded for "
        "or picked up. hot: ESPN ranks him far higher over the last 15 days than before the season. "
        "not_for_sale: untouchable. on_the_block: on his owner's ESPN trade block."
    )
    category: str | None = Field(default=None, description="The category a keystone carries")
    reason: str


class PlayerRef(BaseModel):
    id: int
    name: str
    pro_team: str = Field(
        default="", description="NBA team abbreviation as ESPN writes it (NO, GS, UTAH); empty for a free agent"
    )
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
    games_share: float | None = Field(
        default=None,
        description="Expected share of his team's remaining games he plays once back from any current injury, "
        "from his games-played record. A typical regular plays about 0.73.",
    )
    injury_prone: bool = Field(
        default=False, description="He has missed many games in recent seasons: valued for fewer games"
    )
    games_out: float | None = Field(
        default=None,
        description="Listed out: expected team games he still misses, from how many he has missed in a row",
    )
    expected_back: str | None = Field(
        default=None, description="Listed out: date of his expected first game back, YYYY-MM-DD"
    )
    out_report: str | None = Field(
        default=None, description="Listed out: the news headline games_out rests on, when it reported a timeline"
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


FactKind = Literal[
    "sell_high",
    "buy_low",
    "punt_cost",
    "players_out",
    "consolidate",
    "chasing",
    "roster_squeeze",
    "keystone_lost",
    "vs_pickup",
    "faab",
    "paper_win",
    "punt_mine",
    "punt_theirs",
    "not_rival",
]


class ReasonFact(BaseModel):
    """One reason as data, so the page can draw it (bars, chips) instead of printing a sentence."""

    kind: FactKind
    tone: Literal["for", "against", "note"] = Field(
        description="for: a reason they say yes; against: a reason they may not; note: context for you"
    )
    text: str = Field(description="The same point as a sentence, as in `reasons`")
    player_id: int | None = None
    player_name: str | None = None
    categories: list[str] = Field(default_factory=list)
    owners_value: float | None = Field(default=None, description="What owners pay for the player, in dollars")
    our_value: float | None = Field(default=None, description="What he's worth in this league's categories, in dollars")
    owners_label: str | None = Field(
        default=None, description='owners_value as owners say it: "$38", "a 4th-round pick"'
    )
    our_label: str | None = None
    count: int | None = None
    rank: int | None = Field(default=None, description="chasing: their place in the standings")
    teams: int | None = None
    playoff_spots: int | None = None
    ratio: float | None = Field(default=None, description="paper_win: trade value they get over what they send")


class PitchNeed(BaseModel):
    """A category the deal helps them in, as the pitch names it."""

    category: str
    league_rank: int = Field(description="Their rank in the category now, 1 is best")
    per_game: float | None = Field(default=None, description="What the deal adds a game, counting stats only")


class CategoryDetail(BaseModel):
    category: str
    mine_before: float = Field(description="My weekly projection in this category today (a percentage as 0-1)")
    mine_after: float = Field(description="The same after the deal")
    my_win_before: float = Field(description="My chance to win this category in a week against an average team")
    my_win_after: float
    theirs_before: float = Field(description="The other team's weekly projection today")
    theirs_after: float
    their_win_before: float = Field(description="The other team's weekly chance to win this category")
    their_win_after: float


class PlayoffWeek(BaseModel):
    period: int = Field(description="Matchup period")
    starts_on: str = Field(description="First day, ISO date")
    ends_on: str = Field(description="Last day, ISO date")
    my_cats_before: float = Field(
        description="My expected categories won that week against an average team, on its real games, today"
    )
    my_cats_after: float
    their_cats_before: float
    their_cats_after: float


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
    facts: list[ReasonFact] = Field(default_factory=list, description="`reasons` as data, in the same order")
    pitch_needs: list[PitchNeed] = Field(default_factory=list, description="`pitch` as data")
    playoff_pct_before: float
    playoff_pct_after: float
    partner_playoff_pct_before: float | None = Field(default=None, description="The other team's playoff odds today")
    partner_playoff_pct_after: float | None = Field(
        default=None, description="The other team's playoff odds after the deal: always up, so they see the gain"
    )
    playoff_week_cats_before: float | None = Field(
        default=None,
        description="My expected categories won per fantasy playoff week against an average team, before the deal",
    )
    playoff_week_cats_after: float | None = Field(default=None, description="The same after the deal")
    vision: list[str] = Field(
        default_factory=list,
        description="Vision trades: why the players you get are worth more later than their owner sees now",
    )
    category_changes: dict[str, float] = Field(description="Change in my weekly win % per category")
    category_detail: list[CategoryDetail] = Field(
        default_factory=list,
        description="Both teams, every category: weekly projection and win chance, before and after",
    )
    playoff_weeks: list[PlayoffWeek] = Field(
        default_factory=list, description="Each fantasy playoff week, both teams, before and after"
    )
    finish_before: float | None = Field(
        default=None, description="My projected final place in the standings (average over simulated seasons)"
    )
    finish_after: float | None = None
    partner_finish_before: float | None = None
    partner_finish_after: float | None = None


class MoveDay(BaseModel):
    date: str = Field(description="YYYY-MM-DD, a day left in this week's matchup")
    add_plays: bool = Field(description="The added player has a game that day and is expected to play")
    drop_plays: bool = Field(description="The same for the dropped player")


class MoveTiming(BaseModel):
    """When a pickup should take effect, judged on this week's matchup (analysis/timing.py)."""

    wait: bool = Field(description="Making the move later this week (or next week) beats making it now")
    make_on: str = Field(description="Day (YYYY-MM-DD) to make the move: today unless waiting pays")
    next_week: bool = Field(description="Best left until the next matchup starts: make_on is its first day")
    earliest_on: str = Field(description="Day it takes effect if made now (a waiver claim: when it processes)")
    cats_now: float = Field(description="Change in expected categories won this matchup if made now")
    cats_best: float = Field(description="The same if made on make_on")
    drop_games_kept: int = Field(description="Games the dropped player still plays for you by waiting")
    add_games_missed: int = Field(description="Games of the added player you miss by waiting")
    week_changes: dict[str, float] = Field(
        description="Change in each category's win chance in this week's matchup, made on make_on"
    )
    days: list[MoveDay] = Field(description="The matchup's days left, today first")


class WeekMatchup(BaseModel):
    """This week's matchup, for moves made in it."""

    opponent_team_id: int
    opponent_name: str
    adds_used: int = Field(description="Pickups you have made in this matchup")
    adds_allowed: int | None = Field(description="Pickups the league allows in this matchup; none: no limit")
    close_categories: list[str] = Field(
        description="Categories still close this week on today's rosters: ones a move can steal"
    )


class PickupIdea(BaseModel):
    add: PlayerRef
    drop: PlayerRef
    open_spot_by_trade: bool = Field(
        description="The spot would come from trading `drop` (he has trade value), not from releasing him"
    )
    stream: bool = Field(
        default=False,
        description="A swap for this week's matchup only: my_cats_gain, category_changes and the odds count "
        "this matchup alone, since a player like `drop` can be picked up again next week",
    )
    availability: str
    my_cats_gain: float
    playoff_pct_before: float
    playoff_pct_after: float
    category_changes: dict[str, float]
    timing: MoveTiming | None = Field(
        default=None, description="Best day this week to make it; none without the NBA calendar or a matchup"
    )


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
    playoff_week_cats: float | None = Field(
        default=None,
        description="Expected categories won per fantasy playoff week against an average team, on that week's games",
    )


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
    vision_trades: list[TradeIdea] = Field(
        default_factory=list,
        description="Vision deals: players off to a slow start the projection expects to fade, worth more by the "
        "playoff weeks; you send players who help their owner now",
    )
    pickups: list[PickupIdea]
    season: list[SeasonRow]
    league_rosters: list[TeamRoster] = Field(description="Every other team's players, for picking a trade target")
    faab_remaining: int | None
    week: WeekMatchup | None = Field(default=None, description="None without the NBA calendar or a matchup")
    waiver_priority: int | None = Field(
        default=None,
        description="Your place in the rolling waiver order (1 = first claim), in leagues without FAAB. "
        "A successful claim usually moves you to the back.",
    )
    assumptions: Assumptions
    generated_at: str
    elapsed_ms: int


class ErrorBody(BaseModel):
    code: str
    message: str


class AnalysisJob(BaseModel):
    """A team's analysis running in the background. Poll with the same request until it is done."""

    status: Literal["running", "done", "failed"]
    progress: float = Field(ge=0, le=1, description="Rough share of the work done, 0 to 1")
    stage: str = ""
    report: AnalysisReport | None = None
    error: ErrorBody | None = None


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
    facts: list[ReasonFact] = Field(default_factory=list, description="`reasons` as data, in the same order")
    pitch_needs: list[PitchNeed] = Field(default_factory=list, description="`pitch` as data")
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
    pick_id: str | None = Field(default=None, description="Identifies the deal; changes when the pick is replaced")
    score: float = Field(default=0.0, description="Ranking score: the smaller side's gain times the odds both agree")
    generated_at: str | None = Field(default=None, description="When this deal became the pick")
    checked_at: str | None = Field(default=None, description="When the pick was last checked against the league")


class VoteRequest(LeagueRequest):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    team_id: int = Field(gt=0, description="The side being voted on")
    take: bool = Field(description="True: that team should take the deal")
    pick_id: str | None = Field(default=None, description="The deal being voted on; refused if the pick has changed")
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


class SignUpRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=200)


class GoogleSignIn(BaseModel):
    credential: str = Field(min_length=20, max_length=4096, description="The ID token from Google Identity Services")


class MyLeague(BaseModel):
    league_id: int
    season: int
    name: str
    team_id: int | None = Field(default=None, description="The team this account is the verified manager of")


class Notice(BaseModel):
    id: int
    text: str
    at: str


class Me(BaseModel):
    id: int
    email: str | None
    has_google: bool
    leagues: list[MyLeague]
    notices: list[Notice]


class AuthResult(BaseModel):
    token: str = Field(description="Send as 'Authorization: Bearer <token>'")
    me: Me


class AuthConfig(BaseModel):
    google_client_id: str | None = Field(default=None, description="Set when Google sign-in is turned on")


class OverviewTeam(BaseModel):
    id: int
    name: str
    abbrev: str
    record: str
    has_manager: bool = Field(description="Some account has verified it")
    mine: bool = False
    wins: int = 0
    losses: int = 0
    ties: int = 0
    seed: int | None = Field(default=None, description="ESPN's current standings place, 1 = first")
    games_back: float | None = Field(default=None, description="Category wins behind the leader")


class CategoryScore(BaseModel):
    stat: str = Field(description="Category code, e.g. PTS or FG%")
    value: float = Field(description="Running total this week; a fraction for FG% and FT%")
    result: str = Field(description="W, L or T for this side so far")


class WeekGames(BaseModel):
    today: int = Field(description="Games today (US date) by rostered players outside IR, started or not")
    left: int = Field(description="Games this week that have not tipped off yet")
    total: int = Field(description="Every game this week")


class MatchupSide(BaseModel):
    team_id: int
    wins: int
    losses: int
    ties: int
    categories: list[CategoryScore] = Field(default_factory=list, description="Empty when ESPN sent none")
    games: WeekGames | None = Field(default=None, description="None when rosters or the NBA calendar are unavailable")


class ProjectedCategory(BaseModel):
    stat: str
    mine: float = Field(description="Your projected total by the last day of the week (percentages as fractions)")
    theirs: float
    winner: Literal["me", "them"] = Field(description="Who is projected to take the category")
    close: bool = Field(description="Close to a coin flip: could go either way")


class BenchedPlayer(BaseModel):
    id: int
    name: str


class CrowdedDay(BaseModel):
    date: str = Field(description="ISO date (US)")
    playing: int = Field(description="Healthy players with a game that day")
    slots: int = Field(description="Lineup slots")
    sit: list[BenchedPlayer] = Field(description="Players with a game who don't make the best lineup")


class MatchupProjection(BaseModel):
    opponent_id: int
    wins: int = Field(description="Categories you're projected to take")
    losses: int
    categories: list[ProjectedCategory]
    my_crowded: list[CrowdedDay] = Field(description="Your days left with more players than slots")
    their_crowded: list[CrowdedDay]


class OverviewMatchup(BaseModel):
    home: MatchupSide
    away: MatchupSide


class MvpCandidate(BaseModel):
    team_id: int
    opponent_id: int
    wins: int
    losses: int
    ties: int
    season_record: str


class LeaguePulse(BaseModel):
    faab_leader_id: int | None = None
    faab_leader_spent: int | None = None
    faab_budget: int | None = None
    claims_last_7_days: int = 0
    trades_this_season: int = 0
    close_matchups: int = Field(default=0, description="Matchups one category or less apart")


class LeagueOverview(BaseModel):
    league_id: int
    season: int
    name: str
    matchup_period: int
    week_start: str | None = Field(default=None, description="Monday of this matchup week, ISO date")
    week_end: str | None = Field(default=None, description="Sunday of this matchup week, ISO date")
    playoff_teams: int | None = None
    teams: list[OverviewTeam]
    matchups: list[OverviewMatchup]
    mvp: MvpCandidate | None = Field(default=None, description="The team with the best category record this week")
    pulse: LeaguePulse
    my_team_id: int | None = None


class ShopRequest(AnalyzeRequest):
    offer_player_ids: list[int] = Field(
        min_length=1, max_length=4, description="Your players to shop. Each deal sends one or more of them."
    )
    faab_max: int | None = Field(
        default=None,
        ge=0,
        description="Most FAAB dollars to add to one deal. Default: 15% of what you have left. 0 turns it off.",
    )


class ShopDeal(TradeIdea):
    kind: Literal["win_win", "paper_win"] = Field(
        description="win_win: their build improves too. paper_win: they win on trade value, you win on the court"
    )
    uses_all: bool = Field(description="The deal sends every player you picked")
    faab: int = Field(default=0, description="FAAB dollars you send with the players (0 = none)")
    target_gain: float = Field(description="Change in your weekly win chance summed over the target categories")


class ShopTarget(BaseModel):
    category: str
    win_pct: float = Field(description="Your weekly win chance in it today")
    reason: Literal["swing", "carried"] = Field(
        description="swing: close to a coin flip each week. carried: the players you're shopping supply a lot of it"
    )


class ShopTeam(BaseModel):
    team_id: int
    team_name: str
    record: str
    deals_found: int = Field(description="Packages with this team that pass the checks, before the season sim")
    best: ShopDeal | None = Field(default=None, description="The best deal with this team, if one holds up")


class ShopReport(BaseModel):
    team_id: int
    team_name: str
    offered: list[PlayerRef]
    playoff_pct_before: float
    faab_remaining: int | None = Field(description="Your FAAB left, or null if the league has no FAAB budget")
    faab_cap: int = Field(description="Most FAAB the search added to one deal")
    targets: list[ShopTarget] = Field(description="Categories the search aims at, picked from your team and players")
    deals: list[ShopDeal] = Field(description="Best deals across the league, best first")
    teams: list[ShopTeam] = Field(description="Every other team, best market first")


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


class InjuryImpactRequest(BaseModel):
    player_id: int = Field(gt=0, description="The player who is out")
    league_id: int | None = Field(default=None, gt=0, description="Tag each teammate with his fantasy team here")
    season: int | None = Field(default=None, description="ESPN season year, e.g. 2027 for 2026-27")


class CategoryGains(BaseModel):
    pts: float
    reb: float
    ast: float
    stl: float
    blk: float
    tpm: float
    to: float
    fg_impact: float = Field(description="FG makes above the league rate per game: positive helps FG%")
    ft_impact: float = Field(description="FT makes above the league rate per game: positive helps FT%")


class Beneficiary(BaseModel):
    player_id: int
    name: str
    owner_team_id: int | None = Field(default=None, description="Fantasy team in the asked league; none = free agent")
    minutes_before: float = Field(
        description="Minutes per game with the injured player playing (recent games count more)"
    )
    minutes_after: float = Field(description="Expected minutes per game while he is out")
    minutes_change: float
    error: float = Field(description="80% band around the measured change; 0 when no games without him were seen")
    games_without: float = Field(
        description="Effective games of evidence without him: older games and last season count less"
    )
    gains: CategoryGains = Field(description="Expected change per game in each category")
    value: float = Field(description="Change in 9-category z-score per game")
    confidence: Literal["strong", "some", "estimate"] = Field(
        description="strong: 8+ games without him and a clear change; some: 3+ games; estimate: mostly the prior"
    )
    note: str | None = None


class InjuryImpact(BaseModel):
    player_id: int
    name: str
    pro_team: str
    minutes: float = Field(description="His recent minutes per game")
    games_missed: float = Field(description="Games his team played without him, this season and last")
    beneficiaries: list[Beneficiary] = Field(description="Teammates who gain, best first; at most 6")
    method: str
    track_record: str = Field(description="How the method did on past seasons' absences")


class ScoresRequest(LeagueRequest):
    day: int | None = Field(
        default=None, ge=1, le=250, description="ESPN scoring period (a day). Default: the last day a game started"
    )
    team_id: int | None = Field(default=None, gt=0, description="Your team: this week's fixtures name your players")
    with_sits: bool = Field(
        default=True,
        description="Work out who sits on crowded days (slow on a cold server). "
        "False leaves ``sits`` empty; ask /api/scores/sits for them on their own",
    )


class SitsRequest(LeagueRequest):
    team_id: int = Field(gt=0)


class DaySits(BaseModel):
    day: int
    date: str | None = None
    sits: list[int] = Field(description="Your players with a game that day whom the best lineup leaves out")


class WeekSits(BaseModel):
    days: list[DaySits] = Field(description="Each day left this week, as in ScoreDay.week")


class GameLine(BaseModel):
    player_id: int
    name: str
    pro_team_id: int
    owner_team_id: int | None = Field(default=None, description="Fantasy team in this league; none = free agent")
    minutes: int
    pts: int
    reb: int
    ast: int
    stl: int
    blk: int
    tpm: int
    to: int
    fgm: int
    fga: int
    ftm: int
    fta: int


class GameSide(BaseModel):
    pro_team_id: int
    abbrev: str
    name: str
    score: int | None = Field(default=None, description="None before tip-off")


class Game(BaseModel):
    id: int
    start: str = Field(description="UTC start time, ISO 8601")
    status: Literal["scheduled", "live", "final"]
    clock: str | None = Field(default=None, description='Live games only: "Q4 2:31", "Half", "End Q3", "OT 0:48"')
    away: GameSide
    home: GameSide
    lines: list[GameLine] = Field(description="Everyone who played, most minutes first")


class FixturePlayer(BaseModel):
    player_id: int
    name: str
    injury_status: str
    pro_team_id: int | None = Field(default=None, description="His NBA team, so the other side is the opponent")


class Fixture(BaseModel):
    id: int
    start: str = Field(description="UTC start time, ISO 8601")
    away: GameSide
    home: GameSide
    my_players: list[FixturePlayer] = Field(description="Your players in this game; empty without team_id")


class FixtureDay(BaseModel):
    day: int
    date: str | None = None
    games: list[Fixture]
    sits: list[int] = Field(
        default_factory=list,
        description="Your players with a game who sit: more of them play than the lineup has slots, "
        "and the best lineup leaves these out. Empty when everyone fits or without team_id",
    )


class ScoreDayRef(BaseModel):
    day: int
    date: str | None = None
    games: int


class ScoreDay(BaseModel):
    league_id: int
    season: int
    day: int
    date: str | None = Field(default=None, description="US date of the games")
    games: list[Game]
    days: list[ScoreDayRef] = Field(description="Every day of the season with games, for the date picker")
    live: bool = Field(description="A game is on; ask again in about a minute")
    week: list[FixtureDay] = Field(
        description="Games still to tip off this week (Monday to Sunday), whatever day is shown"
    )
    updated_at: str
    complete: bool = Field(
        default=True,
        description="Scores and box scores are in. False from /api/scores/games before the day's stats are "
        "fetched: started games have no score or lines yet",
    )


class NewsRequest(BaseModel):
    league_id: int | None = Field(default=None, gt=0, description="Tag each player with his fantasy team here")
    season: int | None = Field(default=None, description="ESPN season year, e.g. 2027 for 2026-27")
    include_held: bool = Field(default=False, description="Also return stories only seen on social sites")


NewsKind = Literal["injury", "return", "trade", "rumor", "signing", "waive", "role", "other"]


class NewsPlayer(BaseModel):
    player_id: int
    name: str
    pro_team: str = Field(description="NBA team abbreviation; empty for a free agent")
    injury_status: str = Field(description="ESPN's current status, e.g. ACTIVE, DAY_TO_DAY, OUT")
    owner_team_id: int | None = Field(default=None, description="Fantasy team in the asked league; none = free agent")


class NewsSourceLink(BaseModel):
    source_id: str
    name: str
    kind: Literal["fact", "outlet", "tip"]
    title: str
    url: str
    published_at: str
    summary: str = ""


class NewsStory(BaseModel):
    id: str
    kind: NewsKind
    status: Literal["confirmed", "reported", "held", "didnt_happen"] = Field(
        description="confirmed: ESPN data or two independent outlets agree. reported: one outlet. "
        "held: social sites only. didnt_happen: a reported trade with no team change after 3 days"
    )
    headline: str = Field(description="The lead source's own headline; never written by us")
    summary: str
    published_at: str = Field(description="First seen, UTC")
    updated_at: str = Field(description="Latest source, UTC")
    players: list[NewsPlayer]
    sources: list[NewsSourceLink] = Field(description="Every source for this story, oldest first")
    severity: int = Field(default=0, description="Injuries: 1 day-to-day, 2 out short term, 3 weeks or more")
    conflict: str | None = Field(default=None, description="Set when sources disagree on how bad an injury is")


class NewsSourceHealth(BaseModel):
    source_id: str
    name: str
    kind: Literal["fact", "outlet", "tip"]
    ok: bool | None = Field(default=None, description="None until the source has been tried")
    last_success: str | None = None
    last_attempt: str | None = None
    items: int = 0
    error: str | None = None


class TrendingPlayer(BaseModel):
    player_id: int
    name: str
    pro_team: str
    injury_status: str
    percent_owned: float
    percent_change: float = Field(description="Change in % of ESPN leagues rostering him, today")
    owner_team_id: int | None = None


class NewsPage(BaseModel):
    league_id: int | None = None
    stories: list[NewsStory]
    sources: list[NewsSourceHealth]
    trending: list[TrendingPlayer]
    updated_at: str | None = Field(default=None, description="Last time every source was checked, UTC")
    refreshing: bool = Field(description="A check of every source is running; ask again in a minute")


class PlayerCardRequest(BaseModel):
    league_id: int = Field(gt=0)
    season: int | None = Field(default=None, description="ESPN season year, e.g. 2027 for 2026-27")
    player_id: int = Field(gt=0)
    core: bool = Field(
        default=False,
        description="Only what the card shows first: no news, and names in moves only when already known. "
        "Fetch /api/player-card/extras alongside for the rest.",
    )


class GameStats(BaseModel):
    minutes: float
    pts: float
    reb: float
    ast: float
    stl: float
    blk: float
    tpm: float
    to: float
    fgm: float
    fga: float
    ftm: float
    fta: float


class PlayerGame(BaseModel):
    game_id: int
    season: int
    day: int = Field(description="Scoring period (day of the season)")
    date: str | None = Field(default=None, description="Tip-off, UTC; none when the calendar lacks the game")
    opponent: str = Field(description="Opponent's abbreviation; empty when unknown")
    home: bool | None = None
    stats: GameStats


class PlayerSeasonLine(BaseModel):
    season: int
    games: int
    per_game: GameStats
    fg_pct: float | None = None
    ft_pct: float | None = None


class PlayerMoveOther(BaseModel):
    player_id: int
    name: str
    kind: Literal["add", "drop", "trade"]
    to_team_id: int = Field(description="0 = free agency")


class PlayerMove(BaseModel):
    id: str
    kind: Literal["draft", "add", "drop", "trade"]
    via: Literal["draft", "waiver", "free_agent", "trade"]
    at: str = Field(description="When ESPN processed it, UTC")
    from_team_id: int = Field(description="0 = free agency")
    to_team_id: int = Field(description="0 = free agency")
    bid: int | None = Field(default=None, description="FAAB paid on a waiver claim; price in an auction draft")
    pick: int | None = Field(default=None, description="Overall pick, on a draft")
    others: list[PlayerMoveOther] = Field(
        default_factory=list, description="Other players in the same move, e.g. the drop that made room"
    )


class PlayerNews(BaseModel):
    id: str
    kind: NewsKind
    status: Literal["confirmed", "reported", "held", "didnt_happen"]
    headline: str
    published_at: str
    updated_at: str
    source: str = Field(description="The latest source's name")
    url: str
    severity: int = 0


class PlayerCard(BaseModel):
    player_id: int
    name: str
    position: str = Field(description="Main position, e.g. PG")
    pro_team: str = Field(description="NBA team abbreviation; empty for a free agent")
    injury_status: str = Field(description="ESPN's current status, e.g. ACTIVE, DAY_TO_DAY, OUT")
    owner_team_id: int | None = Field(default=None, description="Fantasy team in this league; none = free agent")
    on_waivers: bool = False
    percent_owned: float
    season_line: PlayerSeasonLine | None = None
    games: list[PlayerGame] = Field(description="Latest games he played, newest first; last season's when none yet")
    news: list[PlayerNews] = Field(description="Recent stories about him, newest first")
    moves: list[PlayerMove] = Field(description="Every move of him in this league this season, newest first")


class PlayerCardExtras(BaseModel):
    """The slower part of a player card, loaded while the core shows."""

    news: list[PlayerNews] = Field(description="Recent stories about him, newest first")
    names: dict[int, str] = Field(description="Names of the other players in his moves, by player id")


FeedbackRating = Literal["bad", "good"]
FeedbackReason = Literal["fake_gain", "wont_accept", "bad_for_build", "player_concern", "already_tried", "other"]
FeedbackSource = Literal["trades", "edge", "shop", "trade_for", "trade_of_day"]


class TradeFeedbackRequest(LeagueRequest):
    team_id: int = Field(gt=0)
    source: FeedbackSource = Field(description="Which list the idea was shown in")
    rating: FeedbackRating
    reasons: list[FeedbackReason] = Field(default_factory=list, max_length=6, description="Quick reasons, any order")
    text: str = Field(default="", max_length=500, description="What the manager wrote, in their own words")
    idea: TradeIdea = Field(description="The idea exactly as it was shown")


class TradeFeedback(BaseModel):
    id: int
    source: FeedbackSource
    rating: FeedbackRating
    reasons: list[FeedbackReason]
    text: str
    idea: TradeIdea
    created_at: str = Field(description="When it was given (or last changed), UTC")


class TradeFeedbackList(BaseModel):
    feedback: list[TradeFeedback] = Field(description="This manager's feedback in the league, newest first")


class TradeFeedbackListRequest(LeagueRequest):
    team_id: int = Field(gt=0)


class TradeCheckRequest(LeagueRequest):
    team_id: int = Field(gt=0)
    idea: TradeIdea = Field(description="The idea as it was shown")
    reasons: list[FeedbackReason] = Field(
        default_factory=list, max_length=6, description="Quick reasons: their checks run first"
    )
    keep_player_ids: list[int] = Field(default_factory=list, description="Players never suggested as drops")


class CheckVerdict(BaseModel):
    check: str = Field(description="free_pickup, noise, lateral_swap, build_conflict, small_sample, injury, ...")
    title: str
    status: Literal["confirmed", "not_supported", "cant_check"]
    evidence: str = Field(description="What the data shows, in a sentence")


class NoteRead(BaseModel):
    kind: Literal["clear", "rant", "unclear"] = Field(
        description="clear: maps onto our reasons and checks; rant: venting; unclear: a point no check covers"
    )
    summary: str = Field(description="What the note says, in one sentence")


class TradeCheck(BaseModel):
    playoff_pct_before: float | None = Field(default=None, description="Today's odds, None when the idea is stale")
    playoff_pct_after: float | None = None
    stale: str | None = Field(default=None, description="Why the idea can't be rebuilt from today's rosters")
    verdicts: list[CheckVerdict]
    read_as: NoteRead | None = Field(default=None, description="How the manager's note was read, when it was")


TradeRuleKind = Literal["protect_category", "skip_team", "no_injury_prone"]


class TradeRuleRequest(LeagueRequest):
    team_id: int = Field(gt=0)
    kind: TradeRuleKind
    value: str = Field(
        default="", max_length=40, description="Category name for protect_category, team id for skip_team, else empty"
    )
    team_name: str = Field(default="", max_length=60, description="For skip_team: the team's name, for the label")
    why: str = Field(default="", max_length=200, description="Where the rule came from, in a sentence")


class TradeRule(BaseModel):
    id: int
    kind: TradeRuleKind
    value: str
    label: str = Field(description="How the rule reads, e.g. 'Protect TO'")
    why: str
    created_at: str


class TradeRuleList(BaseModel):
    rules: list[TradeRule]


class TradeRuleListRequest(LeagueRequest):
    team_id: int = Field(gt=0)


class TradeRuleDeleteRequest(LeagueRequest):
    team_id: int = Field(gt=0)
    rule_id: int = Field(gt=0)


class RecapRequest(LeagueRequest):
    period: int | None = Field(
        default=None, ge=1, le=40, description="Matchup period (week). Default: the last week that has finished"
    )


class WeekTotals(BaseModel):
    """A player's totals over the week, in the site's stat order."""

    minutes: int
    pts: int
    fgm: int
    fga: int
    ftm: int
    fta: int
    tpm: int
    reb: int
    ast: int
    stl: int
    blk: int
    to: int


class RecapPlayer(BaseModel):
    player_id: int
    name: str
    pro_team_id: int
    games: int
    value: float = Field(description="Sum of the week's category z-scores against the 200 busiest players")
    rank: int = Field(description="Rank by value among every NBA player that week")
    z: dict[str, float] = Field(description="Category name -> z-score; turnovers count against")
    totals: WeekTotals


class RecapTopPlayer(RecapPlayer):
    owner_team_id: int | None = Field(default=None, description="Fantasy team on the week's last day; none = free")


class RecapAfter(BaseModel):
    """What a player did for the team that added him, from the day it added him to the week's end."""

    games: int
    started: int = Field(description="Of those games, how many he started (not on the bench or IR)")
    value: float = Field(description="The same 9-cat value as the week's, for these games only")
    totals: WeekTotals


class RecapPickup(BaseModel):
    team_id: int
    day: int = Field(description="Scoring period he first appeared on that roster")
    weekday: int = Field(description="0 = the week's first day (Monday), 6 = Sunday")
    after: RecapAfter = Field(
        description="His games for that team after the pickup; his week's rank counts all of them"
    )


class RecapFreeAgent(RecapPlayer):
    picked_up: RecapPickup | None = Field(default=None, description="Who added him during the week, if anyone")
    owner_now_team_id: int | None = Field(default=None, description="Fantasy team today; none = still free")


class RecapAdd(RecapPlayer):
    day: int
    weekday: int
    after: RecapAfter = Field(description="His games for this team after the pickup, which is what ranks the adds")


class RecapBench(BaseModel):
    games: int = 0
    pts: int = 0
    reb: int = 0
    ast: int = 0
    stl: int = 0
    blk: int = 0
    tpm: int = 0


class RecapTeam(BaseModel):
    team_id: int
    abbrev: str = ""
    name: str = ""
    mvp: RecapPlayer | None = Field(default=None, description="Best player among the games the team started")
    best_add: RecapAdd | None = Field(
        default=None, description="The add who did the most for the team after it added him, this week"
    )
    bench: RecapBench = Field(description="Games played while on the bench or IR, and what they scored")


class RecapSide(BaseModel):
    team_id: int
    wins: int
    losses: int
    ties: int
    values: dict[str, float] = Field(description="Category name -> the week's total (ratios as fractions)")


class RecapMatchup(BaseModel):
    home: RecapSide
    away: RecapSide
    winner_team_id: int | None = Field(default=None, description="None = tied")
    margin: int = Field(description="Category wins between the two sides")


class RecapWeekRef(BaseModel):
    period: int
    start: str | None = None
    end: str | None = None


class WeekRecap(BaseModel):
    league_id: int
    season: int
    period: int
    start: str | None = Field(default=None, description="First day's US date")
    end: str | None = Field(default=None, description="Last day's US date")
    categories: list[str] = Field(description="The league's categories, in its order")
    matchups: list[RecapMatchup] = Field(description="Biggest margin first")
    players: list[RecapTopPlayer] = Field(description="Players of the week, best first")
    free_agents: list[RecapFreeAgent] = Field(description="Best players who were free when the week began")
    teams: list[RecapTeam]
    weeks: list[RecapWeekRef] = Field(description="Every finished week, for a week picker")

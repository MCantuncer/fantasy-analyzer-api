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


class NextGameStats(BaseModel):
    """A per-game line: minutes and the nine categories' stats."""

    minutes: float
    pts: float
    reb: float
    ast: float
    stl: float
    blk: float
    three_pm: float
    to: float
    fgm: float
    fga: float
    ftm: float
    fta: float
    fg_pct: float | None = Field(default=None, description="FGM / FGA; none without attempts")
    ft_pct: float | None = Field(default=None, description="FTM / FTA; none without attempts")


class NextGameReason(BaseModel):
    """One thing about the game that moves his line away from his usual one (analysis/gamebox.py)."""

    kind: Literal[
        "opponent",
        "form",
        "teammates_out",
        "teammates_back",
        "back_from_absence",
        "heavy_minutes",
        "recent_minutes",
        "starter_switch",
        "coach_decision",
        "home",
        "away",
        "favored",
        "underdog",
        "back_to_back",
        "new_team",
        "season_end",
        "lopsided_games",
        "blowout",
        "overtime",
        "rotation",
    ]
    text: str = Field(description="What it is, in a few words: 'Against PHX', 'James out'")
    minutes: float = Field(description="Minutes it adds (+) or takes (-)")
    pts: float = 0.0
    reb: float = 0.0
    ast: float = 0.0
    stl: float = 0.0
    blk: float = 0.0
    three_pm: float = 0.0
    to: float = 0.0
    fg_pct: float = Field(default=0.0, description="Change in FG%, as a share (+0.012 = 1.2 points of FG%)")
    value: float = Field(description="Its 9-category value a game (TO counted against), to rank reasons by")


class NextGameLine(NextGameStats):
    """His line for his next game: the per-game line this week's odds count (projection, his last 7
    and 15 days of minutes, minutes a teammate's absence gives him), with points, rebounds, blocks and
    turnovers set to the level players reach (report.NEXT_GAME_LEVEL), then the game itself
    (analysis/gamebox.py): the opponent's defense in each stat, his recent form (recent games count most,
    last season's end included), whether he started his last game, teammates ruled out, games he just
    missed, home court, how favored his team is and back-to-backs, each tested on five seasons
    (scripts/next_game_matchup_backtest.py)."""

    opponent: str | None = Field(default=None, description="The opponent's abbreviation, when the game is known")
    home: bool | None = Field(default=None, description="True at home, False away, None when the game isn't known")
    tip: str | None = Field(default=None, description="Tip-off, ISO 8601 UTC, when the game and its time are known")
    usual: NextGameStats | None = Field(
        default=None, description="His line on a plain night (no opponent, form, absences): what the reasons move from"
    )
    reasons: list[NextGameReason] = Field(default_factory=list, description="Biggest first, by value")


class HotRun(BaseModel):
    """What a hot run over his last 15 days was made of, and how much of it tends to last (analysis/form.py)."""

    verdict: Literal["straw_fire", "bigger_role", "mixed"] = Field(
        description="bigger_role: minutes are half the gap or more (that part tends to stick). straw_fire: "
        "minutes are a quarter or less, the rest is hot shooting and more shots per minute (fades). mixed: between"
    )
    games: int = Field(description="Games he played in the last 15 days")
    gap: float = Field(description="Standard 9-cat value per game above his usual level (0 = average starter)")
    minutes_part: float = Field(description="Of the gap: his usual per-minute line at the run's minutes")
    production_part: float = Field(description="Of the gap: more shots, boards, assists and stocks per minute")
    shooting_part: float = Field(description="Of the gap: FG%, 3P% and FT% above his usual on the same attempts")
    likely_kept: float = Field(description="Share of the gap expected to last over his next 10 games")
    minutes_recent: float
    minutes_usual: float
    pts_recent: float
    pts_usual: float
    fg_recent: float | None = None
    fg_usual: float | None = None
    three_recent: float | None = None
    three_usual: float | None = None
    ft_recent: float | None = None
    ft_usual: float | None = None


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
    hot_run: HotRun | None = Field(default=None, description="Set when his last 15 days run well above his usual")
    next_game: NextGameLine | None = Field(default=None, description="His line for his next game; none while he is out")


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
    pending: list[str] = Field(
        default_factory=list,
        description="Report sections still being worked out (in a preview: trades, edge_trades, vision_trades). "
        "Empty in a finished report.",
    )


class SampleLeague(BaseModel):
    name: str = Field(description='Always "Sample league": the real name is masked')
    size: int
    scoring: str = Field(description="ESPN's scoring type, e.g. H2H_EACH_CATEGORY")
    categories: list[str]
    playoff_teams: int


class SampleTeam(BaseModel):
    id: int
    name: str = Field(description="Masked: a basketball play call (Horns, Hammer, ...), fixed for the season")
    abbrev: str
    projected_rank: int = Field(description="Projected final place, 1 = first, from each team's own season outlook")
    has_move: bool = Field(description="Its best move gains at least 1 playoff-odds point")


class SampleIndex(BaseModel):
    analysed_on: str = Field(description="The day the sample was analysed (UTC date); its teams' reports are from it")
    fresh: bool = Field(description="Analysed today or yesterday; false means today's run is late")
    phase: Literal["season", "offseason"] = Field(
        description="offseason: last season's final sample, until the new season's first run"
    )
    season: int
    league: SampleLeague
    default_team_id: int = Field(description="The team a visitor scouts first: the clearest move of the day")
    teams: list[SampleTeam]


class ErrorBody(BaseModel):
    code: str
    message: str


class ReportChange(BaseModel):
    """Something that moved in the league since a saved report was built."""

    kind: Literal["status", "added", "dropped", "moved", "games"] = Field(
        description="status: an injury tag or IR slot changed; added/dropped/moved: a roster move; "
        "games: new box scores"
    )
    text: str = Field(description="Plain words, e.g. 'Jayson Tatum is out'")
    player_id: int | None = None
    team_id: int | None = Field(default=None, description="The fantasy team the player is on now (or left)")
    mine: bool = Field(default=False, description="True when it touches the team asking")


class AnalysisJob(BaseModel):
    """A team's analysis running in the background. Poll with the same request until it is done."""

    status: Literal["running", "done", "failed"]
    progress: float = Field(ge=0, le=1, description="Rough share of the work done, 0 to 1")
    stage: str = ""
    report: AnalysisReport | None = None
    preview: AnalysisReport | None = Field(
        default=None,
        description="While running: the report so far, once free agents and the season table are in. Its trade "
        "lists are empty and named in its `pending`; the finished report replaces it.",
    )
    saved: AnalysisReport | None = Field(
        default=None,
        description="While running: the last saved report, out of date because of `changes` (or its age). "
        "Show it until the new one is done.",
    )
    changes: list[ReportChange] = Field(
        default_factory=list,
        description="What moved in the league since `saved` was built, the asking team's news first",
    )
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
    gains: list[PitchNeed] = Field(
        default_factory=list,
        description="Every category this team gains 2+ points of weekly win chance in (up to 4), biggest first, "
        "with its rank now and the per-game change",
    )
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
    progress: float | None = Field(
        default=None, ge=0, le=1, description="While pending: rough share of the league scan done, 0 to 1"
    )
    stage: str | None = Field(
        default=None,
        description="While pending: what the scan is doing (Waiting its turn, while the league's analyses run "
        "first; Loading the league; Screening deals; Simulating the season)",
    )


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


class ForgotPasswordRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=20, max_length=200, description="The token from the emailed reset link")
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
    admin: bool = Field(
        default=False, description="An admin account (users.user_type = 'admin'): can read error reports and usage"
    )


class AuthResult(BaseModel):
    token: str = Field(description="Send as 'Authorization: Bearer <token>'")
    me: Me


class AuthConfig(BaseModel):
    google_client_id: str | None = Field(default=None, description="Set when Google sign-in is turned on")
    password_reset: bool = Field(default=False, description="True when reset links can be emailed")


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


class MatchupProjectionRequest(LeagueRequest):
    team_id: int | None = Field(
        default=None, gt=0, description="The team to project. Default: the team this account verified"
    )


class LineupRequest(LeagueRequest):
    team_id: int = Field(gt=0)
    day: int | None = Field(
        default=None, ge=1, le=250, description="ESPN scoring period (a day). Default: the last day a game started"
    )


class SeasonAverage(BaseModel):
    """A player's real per-game line this season (not a projection)."""

    gp: int = Field(description="Games played this season")
    min: float
    fgm: float
    fga: float
    ftm: float
    fta: float
    tpm: float
    reb: float
    ast: float
    stl: float
    blk: float
    to: float
    pts: float


class LineupDay(BaseModel):
    day: int
    date: str | None = None


class NextGame(BaseModel):
    day: int
    date: str | None = None
    start: str = Field(description="Tip-off, UTC ISO 8601")
    opponent: GameSide
    home: bool


class PlayChance(BaseModel):
    """How likely a player with an injury tag is to play his game that day (app/playchance.py)."""

    chance: float = Field(description="0 to 1: share of past players in his spot who played")
    tag: str = Field(
        description='The NBA report\'s tag: "Probable", "Questionable", "Doubtful", "Available", "Out", '
        'or "Day-to-day" when ESPN has no word yet on which'
    )
    basis: str = Field(description="What the number comes from, in words")
    news: str | None = Field(default=None, description="ESPN's latest one-line injury note")
    news_at: str | None = Field(default=None, description="When ESPN dated that note, UTC ISO 8601")
    return_date: str | None = Field(default=None, description="ESPN's expected return date, YYYY-MM-DD")
    upgraded: bool = Field(default=False, description="Listed out by ESPN, but the latest news has him in play")
    missed_in_a_row: int | None = Field(default=None, description="Team games he has missed in a row, when out")


class LineupSpot(BaseModel):
    player_id: int
    name: str
    pro_team_id: int
    slot: str = Field(description='ESPN lineup slot: "PG", "SG", "SF", "PF", "C", "G", "F", "UTIL", "BE", "IR", ...')
    slot_id: int
    starting: bool = Field(description="In a starting slot: only these players' lines count that day")
    injury_status: str
    game_id: int | None = Field(default=None, description="His NBA team's game that day; none = no game")
    eligible: list[str] = Field(
        default_factory=list, description="Starting slots he can fill, so a bench player who plays can be moved up"
    )
    season_avg: SeasonAverage | None = Field(
        default=None, description="His per-game line this season; none before his first game"
    )
    next_game: NextGame | None = Field(default=None, description="His NBA team's first game after this day")
    week_games: list[int] = Field(
        default_factory=list, description="Days of this Monday-to-Sunday week (see DayLineup.week) his NBA team plays"
    )
    play: PlayChance | None = Field(
        default=None,
        description="His chance to play that day's game, for an injury tag and a game not yet started",
    )


class DayLineup(BaseModel):
    league_id: int
    season: int
    team_id: int
    day: int
    date: str | None = None
    spots: list[LineupSpot] = Field(description="Starting slots in ESPN's order, then the bench, then IR")
    open_slots: list[str] = Field(description="Starting slots nobody fills that day, one entry per empty seat")
    week: list[LineupDay] = Field(default_factory=list, description="The Monday-to-Sunday week this day falls in")
    fetched_at: str = Field(description="When this lineup was read from ESPN, UTC ISO 8601")


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
    starter: bool = Field(default=False, description="In his team's starting five")
    on_court: bool | None = Field(
        default=None, description="On the floor right now; set only while the game is live and ESPN says"
    )


class GameSide(BaseModel):
    pro_team_id: int
    abbrev: str
    name: str
    score: int | None = Field(default=None, description="None before tip-off")


class GameCall(BaseModel):
    """Our call for a game (analysis/gamescore.py), from the games before its day: fixed at tip-off."""

    away: int = Field(description="Points we expect the away team to score")
    home: int = Field(description="Points we expect the home team to score; never equal to away")
    chance: float = Field(description="How sure the call is of its winner, 0.5-1 (calibrated on five seasons)")


class Game(BaseModel):
    id: int
    start: str = Field(description="UTC start time, ISO 8601")
    status: Literal["scheduled", "live", "final"]
    clock: str | None = Field(default=None, description='Live games only: "Q4 2:31", "Half", "End Q3", "OT 0:48"')
    away: GameSide
    home: GameSide
    lines: list[GameLine] = Field(description="Everyone who played, most minutes first")
    call: GameCall | None = Field(default=None, description="Our call for the score; None before the box is built")


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


class PregameRequest(LeagueRequest):
    game_id: int = Field(gt=0, description="The game's id, as /api/scores lists it")
    day: int | None = Field(default=None, ge=1, le=250, description="Its day, when known (saves a search)")


class PregameTerm(BaseModel):
    kind: Literal["matchup", "home_court", "three_point_luck", "back_to_back"] = Field(
        description="matchup: pace, offense and defense, and who plays tonight; home_court; three_point_luck: "
        "three-point shooting on both ends that shouldn't last; back_to_back"
    )
    points: int = Field(description="Points of the called margin, from the picked side; the terms add up to it")
    raw: float


class PregameCallPart(BaseModel):
    """One side's expected points, piece by piece (analysis/gamescore.py): they add up to ``call_raw``."""

    ratings: float = Field(description="Possessions x its points a trip x the opponent's allowed / the league's")
    level: float
    home: float = Field(description="+ at home, - on the road")
    opp_three_luck: float = Field(description="The opponent's three-point defense luck that shouldn't last")
    own_three_luck: float = Field(description="Its own three-point shooting luck that shouldn't last")
    back_to_back: float
    lineup: float = Field(
        default=0.0, description="Tonight's lineup's talent against the call (analysis/lineupcall.py)"
    )
    absences: float = Field(
        default=0.0,
        description="Regulars missing tonight (-) or playing more than usual (+) against how often they play",
    )
    possessions: float = Field(description="Trips each this game: its pace x the opponent's / the league's")
    offense: float = Field(description="Its points a trip so far (recent games count most)")
    defense_allowed: float = Field(description="Points a trip the opponent allows")
    league: float = Field(description="The league's points a trip")
    points_a_game: float = Field(description="Its points a game so far (recent games count most)")


class PregamePlayer(BaseModel):
    player_id: int
    name: str
    injury_status: str
    owner_team_id: int | None = Field(default=None, description="His fantasy team in this league; None: free agent")
    in_pool: bool = Field(
        description="The league's own list (rosters and the free agents it reads). False: a rotation player "
        "read for this box only"
    )
    line: NextGameLine


class PregameOut(BaseModel):
    player_id: int
    name: str
    injury_status: str
    owner_team_id: int | None = None


class PregameTeamLine(BaseModel):
    """How the players' lines stand against the team's call (report.team_sum_gap's view)."""

    shown: float = Field(description="The listed players' points lines, added up")
    rest: float = Field(description="The rest of the team: its points a game less the listed players' season points")
    total: float
    call: float = Field(description="The team's expected points before rounding")
    gap: float = Field(description="total - call: + the lines run over Our call, - under it")


class PregameSide(BaseModel):
    pro_team_id: int
    abbrev: str
    name: str
    call: int
    call_raw: float
    parts: PregameCallPart
    players: list[PregamePlayer] = Field(description="Everyone expected to play, most minutes first")
    out: list[PregameOut] = Field(description="Ruled out tonight")
    team_line: PregameTeamLine


class PregameBox(BaseModel):
    league_id: int
    season: int
    day: int
    game_id: int
    start: str = Field(description="Tip-off, UTC ISO 8601")
    status: Literal["scheduled", "live", "final"]
    chance: float = Field(description="How sure Our call is of its winner, 0.5-1")
    pick: Literal["away", "home"]
    margin: int = Field(description="The called margin in whole points, from the picked side")
    margin_raw: float
    terms: list[PregameTerm] = Field(description="What the margin adds up from, in whole points")
    away: PregameSide
    home: PregameSide


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
    published_at: str = Field(
        description="When the story broke: its earliest source's own publish time, UTC. ESPN player data has "
        "no publish time, so its items carry the time a refresh noticed the change"
    )
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


class NewsCheck(BaseModel):
    refreshing: bool = Field(description="A check of every source is running; ask /api/news again in a few seconds")
    updated_at: str | None = Field(default=None, description="Last time every source was checked, UTC")
    next_check_at: str | None = Field(
        default=None, description="Before this time a new check is not started (the news is already this fresh), UTC"
    )


class IndexedPlayer(BaseModel):
    player_id: int
    name: str
    pro_team: str = Field(description="NBA team abbreviation; empty when he is on no NBA team")
    position: str
    percent_owned: float = Field(description="Share of ESPN leagues that roster him, 0-100")
    owner_team_id: int | None = Field(description="His team in this league; null for a free agent")
    injury_status: str | None = Field(
        default=None, description="ESPN injury status (OUT, DAY_TO_DAY...) when known; null when healthy"
    )
    on_roster: bool = Field(
        default=True,
        description="On his NBA team's roster today (ESPN keeps long-gone players on their old team): "
        "rostered in this league, or in ESPN's news in the last 120 days",
    )


class PlayerIndex(BaseModel):
    """Every NBA player this season, for search as you type: the device filters, not the server."""

    league_id: int
    season: int
    players: list[IndexedPlayer]


class NbaTeamRequest(LeagueRequest):
    team: str = Field(min_length=1, max_length=5, description="NBA team abbreviation as ESPN writes it: DEN, GS, UTAH")


class TeamScorer(BaseModel):
    player_id: int
    name: str
    pts: int
    reb: int
    ast: int


class TeamResult(BaseModel):
    game_id: int
    date: str | None = Field(description="US game date, YYYY-MM-DD")
    start: str
    home: bool
    opponent: str = Field(description="Opponent abbreviation")
    opponent_name: str
    points: int
    opponent_points: int
    won: bool
    top: TeamScorer | None = Field(description="This team's top scorer in the game")


class TeamNextGame(BaseModel):
    start: str
    home: bool
    opponent: str
    opponent_name: str


class NbaTeam(BaseModel):
    """An NBA team at a glance: record, latest results and next game. The roster is the player index's."""

    abbrev: str
    name: str
    wins: int
    losses: int
    streak: str | None = Field(description='"W3" for three wins in a row; null before the first game')
    latest: list[TeamResult] = Field(description="The last five results, most recent first")
    next_game: TeamNextGame | None


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
    won: bool | None = Field(
        default=None, description="Whether his team won; none while the game may still be going or when unknown"
    )
    played: bool = Field(default=True, description="False when his team played and he sat out; stats are then all 0")
    missed_reason: str | None = Field(
        default=None,
        description="Why he sat out, when ESPN says (his current injury, e.g. 'Ankle', or 'Suspended'); "
        "only on the games he has missed since he last played",
    )
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
    summary: str = Field(default="", description="The lead source's own summary; never written by us")
    sources: list[NewsSourceLink] = Field(default_factory=list, description="Every source for this story, oldest first")
    conflict: str | None = Field(default=None, description="Set when sources disagree on how bad an injury is")


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
    hot_run: HotRun | None = Field(default=None, description="Set when his last 15 days run well above his usual")


class GameScore(BaseModel):
    team: int = Field(description="His team's final score")
    opp: int = Field(description="The other team's final score")


class GameNote(BaseModel):
    """What ESPN's play-by-play adds to one finished game of his."""

    score: GameScore | None = Field(default=None, description="Final score, his team first")
    garbage: GameStats | None = Field(
        default=None,
        description="His line in garbage time (4th quarter, lead of 25+ with 12-9 min left, 20+ with 9-6, "
        "10+ under 6, two or fewer starters on the floor); all zeros when he had none; none when his line "
        "couldn't be rebuilt from the plays exactly",
    )


class PlayerCardExtras(BaseModel):
    """The slower part of a player card, loaded while the core shows."""

    news: list[PlayerNews] = Field(description="Recent stories about him, newest first")
    names: dict[int, str] = Field(description="Names of the other players in his moves, by player id")
    games: dict[int, GameNote] = Field(
        default_factory=dict, description="Final score and garbage-time line of his finished games, by game id"
    )
    next_game: NextGameLine | None = Field(
        default=None,
        description="His line for his next game, as FA Targets shows it; none while he is out or when this "
        "league doesn't list him",
    )


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


class OfferCheckRequest(AnalyzeRequest):
    partner_team_id: int = Field(gt=0, description="The other team in the trade")
    give_player_ids: list[int] = Field(min_length=1, max_length=3, description="Your players in the trade")
    get_player_ids: list[int] = Field(min_length=1, max_length=3, description="Their players in the trade")
    my_drop_player_ids: list[int] = Field(
        default_factory=list,
        max_length=2,
        description="Who you'd cut to fit the extra players you get; your least valuable droppable players if empty",
    )
    direction: Literal["received", "sending"] = Field(
        default="received", description="received: they offered it to you. sending: you plan to propose it"
    )


class OfferCutOption(BaseModel):
    player: PlayerRef = Field(description="tier says whether he is droppable or has trade value")
    chosen: bool = Field(description="The cut this check used")
    playoff_pct_after: float | None = Field(
        default=None,
        description="Your playoff odds after the trade with this cut; given for the cheapest few of a one-player cut",
    )


class OfferCheck(BaseModel):
    direction: Literal["received", "sending"] = Field(
        description="received: idea.acceptance and idea.pitch are still filled but don't apply, since they offered it"
    )
    team_id: int
    team_name: str
    partner_team_id: int
    partner_team_name: str
    partner_record: str
    call: Literal["take", "close", "pass"] = Field(
        description="For you: take (odds up a point beyond noise), pass (down a point beyond noise), close otherwise"
    )
    call_reason: str = Field(description="Why, in a sentence")
    odds_noise: float = Field(description="Simulation noise on your odds change (two standard errors, 0-1)")
    helps: list[str] = Field(description="Your categories whose weekly win chance rises 2+ points, biggest first")
    hurts: list[str] = Field(description="Your categories whose weekly win chance drops 2+ points, biggest first")
    partner_call: Literal["gain", "even", "lose"] = Field(description="Their playoff odds: up a point, flat, down")
    partner_reason: str
    partner_helps: list[str] = Field(description="Their categories that rise, ignoring the ones they punt")
    partner_hurts: list[str]
    partner_punts: list[str] = Field(description="Categories they win less than 30% of the time")
    my_punts: list[str] = Field(default_factory=list, description="Categories you win less than 30% of the time")
    paper_win: bool = Field(description="They get at least the trade value they send, owner premium included")
    my_drops: list[PlayerRef] = Field(description="Who you cut to fit the extra players you get")
    my_drop_options: list[OfferCutOption] = Field(
        default_factory=list, description="Players you could cut instead, cheapest first (kept players left out)"
    )
    my_adds: list[PlayerRef] = Field(description="Free agents you pick up for the spots the trade opens")
    their_drops: list[PlayerRef] = Field(description="Who they would likely cut")
    their_adds: list[PlayerRef] = Field(description="Free agents they pick up for the spots the trade opens")
    vs_pickup: bool = Field(description="Your odds are measured against picking up your free agents without the trade")
    flags: list[CheckVerdict] = Field(description="Concerns the data confirms, e.g. a category you compete in drops")
    checks: list[CheckVerdict] = Field(
        default_factory=list,
        description="Every data check that ran, confirmed or clear, so a page can name the clear ones",
    )
    idea: TradeIdea = Field(description="The full picture: both teams' odds, categories, playoff weeks, value, pitch")
    sims: int = Field(description="Simulated seasons behind the odds")
    generated_at: str
    elapsed_ms: int


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


class KeepersRequest(LeagueRequest):
    team_id: int = Field(gt=0)


class KeepersSave(KeepersRequest):
    player_ids: list[int] = Field(max_length=40, description="Every player kept on this team; replaces the saved list")


class Keepers(BaseModel):
    """The players an account keeps on its team: never offered in trades or suggested as drops."""

    player_ids: list[int]
    saved: bool = Field(description="False until this account has saved keepers for the team, even an empty list")


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


class PageView(BaseModel):
    """A page the website opened, for the usage numbers. Only the page name: no ids, no query."""

    page: str = Field(max_length=40, description="A tab or screen, e.g. moves, scores, my-leagues")


class ClientError(BaseModel):
    """A crash or unhandled error in the website, sent so it shows up in the error reports."""

    kind: Literal["crash", "error", "rejection"] = Field(
        description="crash: a screen failed to draw; error: an uncaught error; rejection: an unhandled promise"
    )
    name: str = Field(default="Error", max_length=120, description="The error's type, e.g. TypeError")
    message: str = Field(default="", max_length=2000)
    stack: str = Field(default="", max_length=16000)
    page: str = Field(default="", max_length=300, description="Path and tab, e.g. /?tab=moves; no other query")
    build: str = Field(default="", max_length=40, description="The website build that sent it")


class ErrorReport(BaseModel):
    id: int
    source: Literal["api", "web"]
    title: str
    message: str
    stack: str
    route: str = Field(description="The API request or website page it last happened on")
    build: str
    user_agent: str
    count: int
    first_seen: str
    last_seen: str
    resolved: bool = Field(description="Marked fixed and not seen since")


class ErrorReports(BaseModel):
    reports: list[ErrorReport]


class ResolveErrorRequest(BaseModel):
    id: int = Field(gt=0)


class NewsLogRequest(BaseModel):
    days: int = Field(default=7, ge=1, le=366, description="How many days back, by when an item was published")
    tagged: bool = Field(default=False, description="Only items with at least one emotion tag")
    limit: int = Field(default=2000, ge=1, le=10000, description="At most this many, newest first")


class NewsLogRow(BaseModel):
    id: str
    source_id: str
    published: str
    title: str
    summary: str
    kind: NewsKind
    player_ids: list[int]
    tags: list[str] = Field(description="Emotion tags from the item's words (app/news/emotion.py)")


class NewsLog(BaseModel):
    rows: list[NewsLogRow]


class PlayerLinesRequest(BaseModel):
    season: int | None = Field(default=None, description="ESPN season (the year it ends); the current one when unset")
    last: int = Field(default=0, ge=0, le=400, description="Only the last N settled days; 0 for the whole season")
    min_games: int = Field(default=1, ge=0, le=200, description="Only players with at least this many games")


class StatMiss(BaseModel):
    line: float = Field(description="His average line")
    actual: float = Field(description="What he averaged")
    miss: float = Field(description="Average of result less line: + he beat his line")
    swing: float = Field(description="Typical size of a miss either way (root mean square)")


class PlayerLineSummary(BaseModel):
    player_id: int
    name: str
    team: int = Field(description="NBA team id in his last saved game")
    team_abbrev: str
    games: int = Field(description="Games he played with a saved line")
    dnp: int = Field(description="Nights he had a line but his team played without him")
    stats: dict[str, StatMiss] = Field(description="minutes, pts, reb, ast, stl, blk, three_pm, to")


class PlayerLines(BaseModel):
    days: int = Field(description="Settled days read")
    players: list[PlayerLineSummary]


class BookLinesRequest(BaseModel):
    season: int | None = Field(default=None, description="ESPN season (the year it ends); the current one when unset")
    day: int | None = Field(default=None, ge=1, description="NBA day (scoring period); the latest saved when unset")


class BookRecordRequest(BaseModel):
    season: int | None = Field(default=None, description="ESPN season (the year it ends); the current one when unset")
    nights: int = Field(default=30, ge=1, le=200, description="How many of the latest saved nights")


class RecordPicks(BaseModel):
    stat: str = Field(description="three_pm, reb, ast, stl, blk, pts or all")
    odds_from: float
    odds_to: float | None = Field(description="None: and up")
    picks: int = Field(description="Settled rungs with value over 0 by our chance")
    hits: int = Field(description="Of those, how many he reached")
    said: float = Field(description="Our average chance on those picks")
    net: float = Field(description="One unit on each at the book's odds: net units (+ = they paid)")


class RecordBucket(BaseModel):
    side: Literal["ours", "book"]
    chance_from: float
    chance_to: float
    rungs: int
    said: float = Field(description="The side's average chance in this bucket")
    happened: float = Field(description="How often he reached the rung")


class BookRecord(BaseModel):
    season: int
    nights: int = Field(description="Saved nights read")
    rungs: int = Field(description="Settled rungs with both chances")
    ours_score: float | None = Field(description="Brier score of our rung chances (lower is better)")
    book_score: float | None = Field(description="Brier score of the book's rung chances, margin taken out")
    picks: list[RecordPicks]
    calibration: list[RecordBucket]


class BookPlayer(BaseModel):
    name: str = Field(description="As iddaa spells it")
    pts: float | None = Field(default=None, description="iddaa's line: the number he goes past half the time")
    reb: float | None = None
    ast: float | None = None
    stl: float | None = None
    blk: float | None = None
    three_pm: float | None = None


class BookGame(BaseModel):
    game_id: int = Field(description="ESPN game id")
    away: str
    home: str
    tip: str
    checked: str = Field(description="iddaa's last look before tip-off")
    home_chance: float | None = Field(description="iddaa's chance the home team wins, margin taken out")
    margin: float | None = Field(description="The home team's handicap: -5.5 = home by 6 or more")
    total: float | None
    home_pts: float | None
    away_pts: float | None
    ours: GameCall | None = Field(default=None, description="Our call for the game at the same look")
    players: list[BookPlayer]


class BookLook(BaseModel):
    checked: str | None = Field(description="The last time the watch read iddaa")
    ok: bool | None = Field(description="Whether that read worked; None before the first")
    error: str | None
    games: int = Field(description="NBA games iddaa listed then")
    matched: int = Field(description="Of those, how many matched an ESPN game")


class BookLines(BaseModel):
    season: int
    day: int | None
    days: list[int] = Field(description="Days with saved iddaa lines")
    look: BookLook
    games: list[BookGame]


class CompareRow(BaseModel):
    """One number of ours beside iddaa's and the result. ✓ (``*_hit``) when the result lands within
    the stat's band of that number (app/bookcompare.py BANDS), judged the same for both sides."""

    ours: float | None = Field(description="Our number, as it stood before tip-off")
    book: float | None = Field(description="iddaa's line, margin taken out")
    actual: float | None = Field(description="What happened; None until the game is final")
    gap: float | None = Field(description="Ours less iddaa's: + when we expect more")
    ours_hit: bool | None
    book_hit: bool | None
    closer: Literal["ours", "book", "tie"] | None = Field(description="Whose number was nearer the result")


class CompareRung(BaseModel):
    """One of the book's "n+" rungs with both chances (app/bookcompare.py ladder_rungs)."""

    n: int
    odd: float = Field(description="The book's decimal odds")
    book: float = Field(description="The book's chance, its game margin taken out")
    ours: float | None = Field(description="Our chance from our line (five replayed seasons); None without a line")
    value: float | None = Field(
        description="Average return of one unit at these odds if our chance is right: ours x odd - 1"
    )
    hit: bool | None = Field(description="He got there; None until the result, or when he didn't play")


class ComparePlayer(CompareRow):
    name: str
    player_id: int | None = Field(description="ESPN id; None when iddaa's name matched nobody")
    team: str = Field(description="NBA team abbreviation; empty when unknown")
    stat: Literal["three_pm", "reb", "ast", "stl", "blk", "pts"]
    played: bool | None = Field(description="False when his team played without him; None before the result")
    ladder: list[tuple[int, float]] = Field(
        default_factory=list, description='iddaa\'s "n+" rungs: n and the chance he gets there, margin taken out'
    )
    rungs: list[CompareRung] = Field(default_factory=list, description="The same rungs with odds, our chance and value")
    best_value: float | None = Field(default=None, description="The highest value among the rungs (+: worth a look)")
    rung_closer: Literal["ours", "book", "tie"] | None = Field(
        default=None, description="Whose chances read the result better over the rungs (Brier score)"
    )


class CompareMarket(CompareRow):
    market: Literal["winner", "margin", "total", "team_pts"] = Field(
        description="winner: chances for our pick; margin: points by our pick; total; team_pts: one side's points"
    )
    team: str = Field(description="Our pick (winner, margin) or the side (team_pts); empty for total")
    book_team: str = Field(default="", description="winner only: iddaa's favourite")
    won: str = Field(default="", description="winner only: who won, once final")


class CompareGame(BaseModel):
    game_id: int
    away: str
    home: str
    tip: str
    checked: str = Field(description="The last look before tip-off")
    status: Literal["scheduled", "live", "final"]
    away_score: int | None
    home_score: int | None
    markets: list[CompareMarket]
    players: list[ComparePlayer]


class CompareTally(BaseModel):
    stat: str = Field(description="three_pm, reb, ast, stl, blk, pts, winner, margin, total, team_pts")
    lines: int
    compared: int = Field(description="Rows with both numbers")
    median_gap: float | None
    ours_above: int
    judged: int = Field(description="Rows with a result")
    ours_hits: int
    book_hits: int
    ours_closer: int
    book_closer: int
    level: int = Field(description="Results both missed by the same (within 0.1)")
    ours_miss: float | None = Field(description="Average distance of ours from the result")
    book_miss: float | None = Field(description="Average distance of iddaa's from the result")
    rungs: int = Field(default=0, description="Settled player rungs")
    ours_rung_score: float | None = Field(default=None, description="Brier score of our rung chances (lower is better)")
    book_rung_score: float | None = Field(default=None, description="Brier score of the book's rung chances")
    ours_rung_closer: int = Field(default=0, description="Player rows where our rung chances read the result better")
    book_rung_closer: int = Field(default=0, description="Player rows where the book's rung chances did")
    value_picks: int = Field(default=0, description="Settled rungs with value over 0 by our chance")
    value_hits: int = Field(default=0, description="Of those, how many he reached")
    value_return: float = Field(default=0.0, description="One unit on each value pick at the book's odds: net units")


class BookCompare(BaseModel):
    season: int
    day: int | None
    date: str | None
    days: list[int] = Field(description="Days with saved iddaa lines")
    nights: list[ScoreDayRef] = Field(description="The same days with their US date and how many games")
    bands: dict[str, float] = Field(description="How near the result a number must land to count as right")
    look: BookLook
    games: list[CompareGame]
    tally: list[CompareTally]


class UsageRequest(BaseModel):
    days: int = Field(default=30, ge=1, le=366, description="How many days back, today included")


class UsageDay(BaseModel):
    day: str
    visitors: int = Field(description="Distinct visitors that day (a daily hash; never matched across days)")
    signed_in: int = Field(description="Of those, how many were signed in")
    new_accounts: int
    views: dict[str, int] = Field(description="Page views by page, e.g. moves, scores")
    events: dict[str, int] = Field(description="Things done on the server, e.g. signup, analysis")


class UsageReport(BaseModel):
    accounts: int
    verified_teams: int
    followed_leagues: int
    days: list[UsageDay] = Field(description="Most recent first; days with nothing are left out")

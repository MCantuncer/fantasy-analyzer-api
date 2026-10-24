# fantasy-analyzer-api

Backend for the ESPN fantasy basketball analyzer. Give it a league ID and team ID and it returns moves ranked by how much they change your playoff odds: trades the other owner should accept, free-agent pickups with the right drop, your category outlook, every team's build, and a full season simulation.

The web client lives in [fantasy-analyzer-web](https://github.com/MCantuncer/fantasy-analyzer-web).

## Run locally

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

Then open http://localhost:8000/docs, or start the web client.

```sh
uv run pytest       # tests use a recorded league, no network
uv run ruff check .
```

## API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| POST | `/api/league` | `{league_id, season?}` | League name, categories and teams, for the team picker |
| POST | `/api/analyze` | `{league_id, team_id, season?, keep_player_ids?}` | Full analysis report |
| POST | `/api/trade-of-the-day` | `{league_id, season?}` | Today's best win-win trade between any two teams, with poll votes |
| POST | `/api/trade-of-the-day/vote` | `{league_id, season?, date, team_id, take, voter}` | Records a "Would you take it?" vote and returns the updated pick |
| POST | `/api/team-link/start` | `{league_id, season?, team_id}` | A code to set as the team's ESPN abbreviation, and a signed challenge |
| POST | `/api/team-link/verify` | `{challenge}` | Checks the abbreviation on ESPN; returns a session token for that team |
| POST | `/api/team-link/me` | `{league_id, season?}` | The linked team (needs the token) |
| POST | `/api/team-link/unlink` | | Forgets the token |
| POST | `/api/feed` | `{league_id, season?, page?}` | A week of moves (claims, losing bids, adds, drops, trades); with a token also the chat, reactions and reply counts |
| POST | `/api/feed/replies` | `{league_id, season?, target}` | Replies on a move or a post |
| POST | `/api/posts` | `{league_id, season?, body, target?}` | Posts to the chat, or replies when `target` is set |
| POST | `/api/posts/delete` | `{league_id, season?, post_id}` | Deletes your own post |
| POST | `/api/reactions` | `{league_id, season?, target, reaction, on?}` | Toggles `steal`, `smart`, `overpay`, `lol`, or a `won:<team id>` trade vote |
| POST | `/api/target` | `{league_id, team_id, player_id, season?, keep_player_ids?}` | One player's fit, his owner's needs, and the best offer for him |
| POST | `/api/scores` | `{league_id, season?, day?, team_id?}` | A day of NBA games: scores, status and box scores, each player tagged with his fantasy team in this league; plus this week's upcoming games, naming `team_id`'s players in each |
| GET | `/api/health` | | `{ok: true}` |

`season` defaults to the current ESPN season (2026-27 is `2027`). `keep_player_ids` are never offered in trades or suggested as drops.

The schema is in `openapi.json`. Regenerate it after changing `app/schemas.py`:

```sh
uv run python -m scripts.export_openapi openapi.json
```

## League chat and linking a team

ESPN has no login for other sites, so a manager proves a team is theirs by changing its abbreviation on ESPN to a 4-character code from `/api/team-link/start` and calling `/api/team-link/verify`. Only the team's owner and the league manager can edit it, and the API reads it back from ESPN's league data. The code rides in an HMAC-signed challenge that expires after 30 minutes, so nothing is stored until the check passes. Then the device gets a random token, sent as `Authorization: Bearer <token>`; only its SHA-256 hash is stored. No ESPN password or cookie is kept.

Set `SECRET_KEY` (any long random string) so challenges survive a restart. Posts, reactions and links use `DATABASE_URL` like the trade of the day, and live in memory without it.

The feed pages through the season a week (seven scoring periods) at a time. Transactions come from ESPN's `mTransactions2` view, one call per day: past days are cached for six hours, today for a minute.

## NBA scores

ESPN's fantasy API has no scoreboard, but for any day it returns every player's line for each game. `/api/scores` adds those up into team scores and box scores, and takes start times from the NBA calendar. There is no quarter or clock, so a game counts as live from tip-off until ESPN marks its stats official or three hours pass.

A day is the same in every league. While a game is on, the day is fetched at most once a minute for everyone. Once a day's games are over (five hours after the last tip-off), it is saved to `DATABASE_URL` (table `nba_days`) and never fetched again. Other features can read finished games from there instead of calling ESPN.

## Private leagues

ESPN has no API keys for fantasy, so private leagues need the user's `espn_s2` and `SWID` cookies. Send them in the `X-ESPN-S2` and `X-ESPN-SWID` headers. The backend:

- uses them only for that request's read-only calls to ESPN,
- never logs, stores or returns them (tests check this),
- caches private-league data under a hash of the cookies, so one user's league is never served to someone else.

ESPN answers `401` for a private league without cookies, and the API passes that on as `401 {"detail": {"code": "private_league"}}`.

## How the analysis works

- **Projections:** ESPN's season projections, blended with this season's per-game stats as games are played (`app/espn/parse.py`).
- **Weekly model:** each team starts its top healthy players (by z-score value) for about 3.4 games a week. Each category's weekly total is a normal distribution; counting stats use an over-dispersed Poisson variance, percentages use binomial variance on attempts (`app/analysis/model.py`).
- **Trades:** every 1-for-1, 2-for-1 and 1-for-2 with every team (`app/analysis/trades.py`). A deal is kept only if it
    - helps your categories,
    - helps theirs too, under their own build, and
    - gives them at least 85% of the value they send as owners see it: ESPN's rankings and auction values, this league's draft prices, and this season's rank as games are played (`app/analysis/perceived.py`).

  Each deal gets rough acceptance odds and a one-line pitch on why it helps them. Deals are ranked by playoff-odds gain weighted by those odds.
- **Trade of the day:** every pair of teams, scanned with the same finder from both sides (`app/analysis/daily.py`). A deal qualifies only if both owners see fair value and both teams gain in the categories they compete in; the top few must also keep both teams' playoff odds from dropping in the simulation. Deals are ranked by the smaller of the two gains, weighted by the chance both owners say yes. Teams take turns in the spotlight (one a day, fixed order per league), and the spotlight team's best deal wins when it's close to the overall best, so the pick moves around the league. The pick is fixed per league per UTC day. The scan takes a few seconds (longer on the free plan), so the endpoint answers `status: "pending"` after `DAILY_TRADE_WAIT_SECONDS` (default 20) and the client asks again.
- **Trading for one player:** pick any rostered player and get his fit with your build (category changes and playoff odds if you added him), his owner's build, needs and which of your players help them, and up to three offers (`app/analysis/target.py`). Offers use the trade finder's rules, except any gain for you counts and the owner's build only has to stay about even. Packages are one or two of your players for him, or two or three for him plus one of their low-value players. The best offer maximises playoff gain weighted by acceptance; fallbacks are the likeliest to be accepted and the cheapest on paper. If nothing keeps the owner's build even, the best long shots are shown with a note.
- **Pickups:** every healthy free agent against every possible drop, keeping the best drop.
- **Season:** 4,000 simulations of the remaining schedule with the league's real format: each category counts in "H2H each category" leagues, one result per matchup in "most categories" leagues (`app/analysis/sim.py`).

Only ESPN head-to-head category leagues are supported for now.

## Backtest

Replays finished seasons of a public league week by week through the live model, using only what was known on each week's first day (rosters, lineups and box scores before that day), and scores the weekly category odds and the playoff odds against what happened (`app/backtest/`):

```sh
uv run python -m scripts.backtest 744636428 2025 2026
```

The first run of a season downloads it from ESPN into `.cache/backtest/` (a couple of minutes); later runs read the cache. Lower Brier and log loss are better; a spread of 1.0 means the model is exactly as unsure as it should be, above 1 it is overconfident. Keep a model change only if it helps every season.

## Test fixture

`tests/fixtures/chill_primera_2027.json.gz` is a trimmed copy of a public league. Re-record it with:

```sh
uv run python -m scripts.record_fixture 744636428 2027 tests/fixtures/chill_primera_2027.json.gz
```

## Deploy

`render.yaml` deploys the API to [Render](https://render.com): New > Blueprint, pick this repo, then set `CORS_ORIGINS` to the web app's URL (for example `https://fantasy-analyzer-web.vercel.app`). For the trade of the day to stay the same all day and for poll votes to be kept across restarts, also set `DATABASE_URL` to a Postgres connection string (a free [Neon](https://neon.tech) database is plenty; the tables are created on start). Without it, picks and votes live in memory and reset when the server sleeps. The free plan sleeps when idle; switch `plan` to `starter` for an always-on server. Access logs are turned off so request details are never written to Render's logs.

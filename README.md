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
| GET | `/api/health` | | `{ok: true}` |

`season` defaults to the current ESPN season (2026-27 is `2027`). `keep_player_ids` are never offered in trades or suggested as drops.

The schema is in `openapi.json`. Regenerate it after changing `app/schemas.py`:

```sh
uv run python -m scripts.export_openapi openapi.json
```

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
- **Pickups:** every healthy free agent against every possible drop, keeping the best drop.
- **Season:** 4,000 simulations of the remaining schedule with the league's real format: each category counts in "H2H each category" leagues, one result per matchup in "most categories" leagues (`app/analysis/sim.py`).

Only ESPN head-to-head category leagues are supported for now.

## Test fixture

`tests/fixtures/chill_primera_2027.json.gz` is a trimmed copy of a public league. Re-record it with:

```sh
uv run python -m scripts.record_fixture 744636428 2027 tests/fixtures/chill_primera_2027.json.gz
```

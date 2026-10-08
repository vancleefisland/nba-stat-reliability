"""Download NBA player game logs and write data/raw/game_logs.csv in the format the
pipeline expects.

Setup:   pip install nba_api
Run:     python scripts/collect_game_logs.py                    (default seasons)
         python scripts/collect_game_logs.py 2022-23 2023-24    (choose seasons)

Uses one call per season for ALL players (LeagueGameLog) and one for all teams.
`poss` (possessions the player was on the floor for) is ESTIMATED:
    team possessions ~= FGA + 0.44*FTA - OREB + TOV   (standard box-score estimate)
    player poss      ~= team possessions * (player MIN / (team MIN / 5))
This is an approximation; say so in your README.
"""
import sys
import time
from pathlib import Path

import pandas as pd

DEFAULT_SEASONS = ["2021-22", "2022-23", "2023-24", "2024-25"]
OUT_PATH = Path("data/raw/game_logs.csv")
PAUSE_SECONDS = 3      # be polite to the API between calls
RETRIES = 3


def fetch(kind: str, season: str) -> pd.DataFrame:
    """kind: 'P' for player logs, 'T' for team logs. Retries on timeouts/blocks."""
    from nba_api.stats.endpoints import leaguegamelog
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            res = leaguegamelog.LeagueGameLog(
                season=season, season_type_all_star="Regular Season",
                player_or_team_abbreviation=kind, timeout=60)
            time.sleep(PAUSE_SECONDS)
            return res.get_data_frames()[0]
        except Exception as e:  # network errors, rate limits, blocked requests
            last_err = e
            print(f"  {season} ({kind}) attempt {attempt} failed: {e}")
            time.sleep(PAUSE_SECONDS * attempt * 2)
    raise RuntimeError(f"Could not fetch {kind} logs for {season}: {last_err}")


def transform(players: pd.DataFrame, teams: pd.DataFrame, season_label: int) -> pd.DataFrame:
    """Pure function (no network): raw API frames -> pipeline schema."""
    t = teams.copy()
    t["team_poss"] = t["FGA"] + 0.44 * t["FTA"] - t["OREB"] + t["TOV"]
    t = t[["GAME_ID", "TEAM_ID", "team_poss", "MIN"]].rename(columns={"MIN": "team_min"})

    p = players.merge(t, on=["GAME_ID", "TEAM_ID"], how="left", validate="many_to_one")
    p = p[p["MIN"].fillna(0) > 0].copy()            # drop DNPs / zero-minute rows
    p["poss"] = (p["team_poss"] * p["MIN"] / (p["team_min"] / 5)).round().astype("Int64")

    out = pd.DataFrame({
        "player": p["PLAYER_ID"].astype(str),
        "player_name": p["PLAYER_NAME"],
        "season": season_label,
        "game_date": pd.to_datetime(p["GAME_DATE"]).dt.date,
        "fg3m": p["FG3M"], "fg3a": p["FG3A"],
        "ftm": p["FTM"], "fta": p["FTA"],
        "ast": p["AST"], "poss": p["poss"],
    })
    return out.dropna(subset=["poss"])


def main(seasons):
    frames = []
    for s in seasons:
        print(f"Fetching {s} ...")
        players, teams = fetch("P", s), fetch("T", s)
        frames.append(transform(players, teams, int(s[:4])))   # 2023-24 -> 2023
        print(f"  {len(frames[-1]):,} player-games")
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["player", "season", "game_date"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with {len(df):,} rows, "
          f"{df.player.nunique():,} players, seasons {sorted(df.season.unique())}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_SEASONS)
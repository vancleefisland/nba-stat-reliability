"""Loading and validating player game logs."""
import pandas as pd
from . import config as C


def load_game_logs(path: str = C.RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["game_date"])
    missing = [c for c in C.REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Game logs missing required columns: {missing}")
    validate(df)
    return df.sort_values(["player", "season", "game_date"]).reset_index(drop=True)


def validate(df: pd.DataFrame) -> None:
    """Fail loudly on problems rather than silently computing garbage."""
    if df.duplicated(["player", "season", "game_date"]).any():
        raise ValueError("Duplicate (player, season, game_date) rows.")
    for stat, spec in C.STATS.items():
        n, d = spec["num"], spec["den"]
        if (df[[n, d]] < 0).any().any():
            raise ValueError(f"{stat}: negative counts found.")
        if (df[n] > df[d]).any():
            raise ValueError(f"{stat}: numerator exceeds denominator in some rows ({n} > {d}).")

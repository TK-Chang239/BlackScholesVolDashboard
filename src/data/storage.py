"""All disk writes for the data layer. Atomic (tmp + rename), schema-checked."""
import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd

from src.data.base import CHAIN_COLUMNS


def chain_path(snapshot_date: dt.date, root: Path) -> Path:
    return Path(root) / "data" / "chains" / f"{snapshot_date.isoformat()}.parquet"


def chain_exists(snapshot_date: dt.date, root: Path) -> bool:
    return chain_path(snapshot_date, root).exists()


def _atomic_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def write_chain(df: pd.DataFrame, snapshot_date: dt.date, root: Path) -> Path:
    if list(df.columns) != CHAIN_COLUMNS:
        raise ValueError(f"chain schema mismatch: {list(df.columns)} != {CHAIN_COLUMNS}")
    return _atomic_parquet(df, chain_path(snapshot_date, root))


def upsert_underlying(df_new: pd.DataFrame, root: Path) -> Path:
    path = Path(root) / "data" / "underlying.parquet"
    if path.exists():
        merged = pd.concat([pd.read_parquet(path), df_new], ignore_index=True)
    else:
        merged = df_new.copy()
    merged = (merged.drop_duplicates("date", keep="last")
                    .sort_values("date").reset_index(drop=True))
    return _atomic_parquet(merged, path)


def read_underlying(root: Path) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / "data" / "underlying.parquet")


def daily_metrics_path(root: Path) -> Path:
    return Path(root) / "data" / "daily_metrics.parquet"


def read_daily_metrics(root: Path, columns: list[str]) -> pd.DataFrame:
    path = daily_metrics_path(root)
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path).reindex(columns=columns)


def write_daily_metrics(df: pd.DataFrame, root: Path) -> Path:
    if "date" not in df.columns:
        raise ValueError("daily_metrics needs a 'date' column")
    if df["date"].duplicated().any():
        raise ValueError("daily_metrics has duplicate dates")
    return _atomic_parquet(df.sort_values("date").reset_index(drop=True),
                           daily_metrics_path(root))


def write_status(status: dict, root: Path) -> Path:
    path = Path(root) / "docs" / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(tmp, path)
    return path


def write_text(text: str, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
    return path

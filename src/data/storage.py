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


def write_status(status: dict, root: Path) -> Path:
    path = Path(root) / "docs" / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(tmp, path)
    return path

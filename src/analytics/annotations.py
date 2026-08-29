"""P6 annotations: dated notes for the skew time series (SPEC 3 P6)."""
import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

COLUMNS = ["date", "note"]


def load_annotations(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    doc = yaml.safe_load(path.read_text()) or {}
    entries = doc.get("annotations") or []
    rows = []
    for e in entries:
        if not isinstance(e, dict) or "date" not in e or "note" not in e:
            raise ValueError(f"annotation entry needs 'date' and 'note': {e!r}")
        d = e["date"]
        if isinstance(d, dt.datetime):
            d = d.date()
        elif not isinstance(d, dt.date):
            d = dt.date.fromisoformat(str(d))
        rows.append({"date": d, "note": str(e["note"])})
    return pd.DataFrame(rows, columns=COLUMNS)

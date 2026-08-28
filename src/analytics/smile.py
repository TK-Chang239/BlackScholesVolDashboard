"""P2: IV smile by expiry (SPEC 3 P2). Pure function, tidy output.

OTM convention: puts carry the smile below spot, calls at/above spot --
one continuous line per expiry built from the options people actually
trade at each strike.
"""
import pandas as pd

SMILE_COLUMNS = ["expiry", "dte", "strike", "moneyness", "kind", "iv"]


def compute_smile(chain_iv: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    spot = float(chain_iv["spot"].iloc[0]) if len(chain_iv) else float("nan")
    available = chain_iv[["expiry", "dte"]].drop_duplicates().reset_index(drop=True)
    chosen: list = []
    for target in cfg["target_dte"]["smile_expiries"]:
        if available.empty:
            break
        nearest = available.iloc[(available["dte"] - target).abs().idxmin()]["expiry"]
        if nearest not in chosen:
            chosen.append(nearest)
    df = chain_iv[chain_iv["expiry"].isin(chosen)].copy()
    otm = ((df["kind"] == "put") & (df["strike"] < spot)) | (
        (df["kind"] == "call") & (df["strike"] >= spot)
    )
    df = df[otm & df["iv"].notna()].copy()
    df["moneyness"] = df["strike"] / spot
    return (df[SMILE_COLUMNS]
            .sort_values(["expiry", "strike"])
            .reset_index(drop=True))

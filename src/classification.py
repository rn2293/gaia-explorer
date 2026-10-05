"""Load and clean the Gaia measurements for Week 3.

The learning code lives in the notebooks: labels, splits, sigmoid, loss,
gradient descent, model fitting, and evaluation. This file only handles data.
"""
from pathlib import Path

import numpy as np
import pandas as pd


def clean_week3_sample(raw, *, snr_threshold=5.0):
    """Keep finite measurements, reliable positive parallaxes, and unique sources.

    Return the cleaned measurements and row counts. Labels and model training
    are written out in the notebooks so the learning steps remain visible.
    """
    required = ["source_id", "bp_rp", "phot_g_mean_mag", "parallax", "parallax_error"]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not np.isfinite(snr_threshold) or snr_threshold <= 0:
        raise ValueError("Use a positive finite SNR threshold.")
    df = raw.copy()
    counts = {"input": len(df)}
    df = df.dropna(subset=["source_id"])
    # IDs are identifiers, never floating-point features or predictors.
    df["source_id"] = df["source_id"].astype(str)
    numeric = required[1:]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    df = df.loc[np.isfinite(df[numeric]).all(axis=1)].copy()
    counts["finite measurements and source ID"] = len(df)
    df = df.loc[(df.parallax > 0) & (df.parallax_error > 0)].copy()
    df["parallax_snr"] = df.parallax / df.parallax_error
    df = df.loc[np.isfinite(df.parallax_snr) & (df.parallax_snr > snr_threshold)].copy()
    counts["positive parallax and SNR cut"] = len(df)
    df = df.drop_duplicates("source_id").reset_index(drop=True)
    counts["unique sources"] = len(df)
    return df, pd.Series(counts, name="rows").to_frame()


def load_week3_data(csv_path=None, *, use_query_if_missing=True, top_n=10_000,
                    verify_ssl=True):
    """Read Gaia data, fetching the shared W2 cache on first use if needed.

    An explicit CSV path is always respected: a typo must not trigger a query
    or create a replacement file. Set use_query_if_missing=False for offline
    use. TLS verification remains enabled by default for archive queries.
    """
    path = Path(csv_path).expanduser() if csv_path is not None else Path(__file__).resolve().parents[1] / "data/processed/gaia_clean_day1.csv"
    if not path.is_file() and csv_path is None and use_query_if_missing:
        # Reuse W2's fetch/clean/cache workflow. Cached runs never import
        # astroquery, and reading IDs as strings below preserves their digits.
        from .data_load import load_clean_gaia_sample

        load_clean_gaia_sample([path], top_n=top_n, verify_ssl=verify_ssl)
    if not path.is_file():
        raise FileNotFoundError(
            f"No Gaia CSV at {path}. Check DATA_PATH, or use "
            "load_week3_data(use_query_if_missing=True) with no CSV path "
            "to fetch and cache the W2 sample."
        )
    raw = pd.read_csv(path, dtype={"source_id": "string"})
    return clean_week3_sample(raw)

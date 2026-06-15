"""
Load the cleaned Gaia sample used by the learning notebooks.

The notebooks should prefer the cached Day 1 CSV so they remain fast and
offline-friendly. If the cache is missing, this module can query Gaia DR3,
apply the shared cleaning rules, and write the cache for later notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .data_clean import clean_sample


DEFAULT_CLEANED_CSV_CANDIDATES = [
    Path("../data/processed/gaia_clean_day1.csv"),
    Path("data/processed/gaia_clean_day1.csv"),
    Path("gaia-explorer/data/processed/gaia_clean_day1.csv"),
]


def _coerce_paths(paths: Iterable[str | Path]) -> list[Path]:
    return [Path(path) for path in paths]


def _first_existing_path(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def _first_writable_cache_path(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    return next((path for path in paths if path.parent.exists()), paths[0])


def load_clean_gaia_sample(
    path_candidates: Iterable[str | Path] = DEFAULT_CLEANED_CSV_CANDIDATES,
    *,
    top_n: int = 10_000,
    use_query_if_missing: bool = True,
    snr_threshold: float = 5,
    verify_ssl: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """
    Load the cleaned Gaia sample, querying and caching it if needed.

    Args:
        path_candidates: Candidate CSV paths, ordered by preference for the
            current notebook working directory.
        top_n: Number of Gaia rows to query if the cleaned CSV is missing.
        use_query_if_missing: If False, raise FileNotFoundError when no cached
            cleaned CSV exists.
        snr_threshold: Minimum parallax signal-to-noise ratio used by cleaning.
        verify_ssl: Whether Gaia archive TLS certificates should be verified.

    Returns:
        A tuple of (clean_dataframe, csv_path). csv_path is the loaded CSV path
        or the path where the queried and cleaned sample was cached.
    """
    paths = _coerce_paths(path_candidates)
    data_path = _first_existing_path(paths)

    if data_path is not None:
        print(f"Loaded cleaned Gaia data from: {data_path}")
        return pd.read_csv(data_path), data_path

    searched = "\n".join(str(path) for path in paths)
    if not use_query_if_missing:
        raise FileNotFoundError(
            "Could not find the Day 1 cleaned CSV. Run day_01_data_cleaning.ipynb first, "
            "or set use_query_if_missing=True.\n"
            f"Searched:\n{searched}"
        )

    # Import only when needed so cached notebook runs do not load astroquery.
    from .data_fetch import fetch_gaia_sample

    print("Day 1 cleaned CSV was not found. Querying Gaia DR3 instead.")
    print(f"Searched:\n{searched}")

    raw_gaia_df = fetch_gaia_sample(top_n=top_n, verify_ssl=verify_ssl)
    gaia_df = clean_sample(raw_gaia_df, snr_threshold=snr_threshold)

    cache_path = _first_writable_cache_path(paths)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    gaia_df.to_csv(cache_path, index=False)

    print(f"Queried Gaia DR3 and saved cleaned data to: {cache_path}")
    print(f"Rows before cleaning: {len(raw_gaia_df):,}")
    print(f"Rows after cleaning: {len(gaia_df):,}")

    return gaia_df, cache_path

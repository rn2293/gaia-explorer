"""
Clean and engineer features from raw Gaia query results.

Raw Gaia data contains measurement noise and missing values that must be
removed before any analysis. This module applies standard quality cuts
used in stellar astronomy papers.
"""

import numpy as np
import pandas as pd


def clean_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality filters to a raw Gaia DataFrame.

    Filters applied:
        1. parallax > 0         — negative parallax is unphysical; it results from
                                  measurement noise on very distant stars.
        2. parallax / parallax_error > 5  — signal-to-noise ratio cut.
                                  SNR < 5 means the uncertainty is so large that the
                                  derived distance could be off by 25%+, making
                                  absolute magnitude calculations unreliable.
        3. bp_rp is not null    — stars without a color index can't be placed on
                                  the HR diagram.

    Args:
        df: Raw DataFrame from data_fetch.fetch_gaia_sample().

    Returns:
        Filtered copy of df. Original is not modified.
    """
    mask = (
        (df["parallax"] > 0) &
        (df["parallax_error"] > 0) &           # guard against zero/NaN error → inf SNR
        (df["parallax"] / df["parallax_error"] > 5) &
        (df["bp_rp"].notna())
    )
    clean = df[mask].copy()
    print(f"Stars before cleaning: {len(df):,}")
    print(f"Stars after cleaning:  {len(clean):,}  ({100 * len(clean) / len(df):.1f}% kept)")
    return clean


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with two derived columns added.

    Columns added:
        distance_pc   — Distance in parsecs: d = 1000 / parallax_mas
                        (parallax is in milliarcseconds; this converts to parsecs)
        absolute_mag  — Absolute magnitude: M = m - 5*log10(d/10)
                        This is the distance modulus formula. It removes the effect
                        of distance so we can compare the true intrinsic brightness
                        of stars regardless of how far away they are.

    Args:
        df: Cleaned DataFrame (output of clean_sample()).

    Returns:
        New DataFrame with distance_pc and absolute_mag columns added. Input is not modified.
    """
    df = df.copy()

    # 1 parsec = distance at which 1 AU subtends 1 arcsecond
    # parallax in mas → distance in pc: d(pc) = 1000 / parallax(mas)
    df["distance_pc"] = 1000 / df["parallax"]

    # Distance modulus: M = m - 5 * log10(d/10)
    # d/10 normalizes to 10 parsecs — the standard reference distance for absolute magnitude
    df["absolute_mag"] = df["phot_g_mean_mag"] - 5 * np.log10(df["distance_pc"] / 10)

    return df

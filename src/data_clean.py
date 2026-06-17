"""
Clean and engineer features from raw Gaia query results.

Raw Gaia data contains measurement noise and missing values that must be
removed before any analysis. This module applies standard quality cuts
used in stellar astronomy papers.
"""

import numpy as np
import pandas as pd


REQUIRED_GAIA_COLUMNS = [
    "parallax",
    "parallax_error",
    "phot_g_mean_mag",
    "bp_rp",
    "ra",
    "dec",
]


def clean_sample(df: pd.DataFrame, *, snr_threshold: float = 5) -> pd.DataFrame:
    """
    Apply quality filters to a raw Gaia DataFrame.

    Filters applied:
        1. Required Gaia columns are present and non-null.
        2. parallax > 0         - negative parallax is unphysical; it results from
                                  measurement noise on very distant stars.
        3. parallax_error > 0   - zero or negative uncertainty makes SNR invalid.
        4. parallax_snr > threshold - low-SNR distances make absolute magnitudes noisy.

    Args:
        df: Raw DataFrame from data_fetch.fetch_gaia_sample().
        snr_threshold: Minimum parallax signal-to-noise ratio to keep.

    Returns:
        Filtered copy of df with parallax_snr added. Original is not modified.
    """
    missing_columns = [col for col in REQUIRED_GAIA_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required Gaia columns: {missing_columns}")

    clean = df.dropna(subset=REQUIRED_GAIA_COLUMNS).copy()
    clean = clean[
        (clean["parallax"] > 0) &
        (clean["parallax_error"] > 0)
    ].copy()
    clean["parallax_snr"] = clean["parallax"] / clean["parallax_error"]
    clean = clean[clean["parallax_snr"] > snr_threshold].copy()

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


def add_absolute_magnitude_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df with the Week 2 regression target columns added.

    Columns added:
        distance_pc - Distance in parsecs, computed from Gaia parallax in mas.
        abs_g_mag   - Absolute Gaia G magnitude, the Week 2 regression target.

    The target formula is the distance modulus written in Gaia parallax form:
        M = m - 5 * log10(d / 10)
        d = 1000 / parallax_mas
        M = m + 5 * log10(parallax_mas) - 10

    Args:
        df: Cleaned Gaia DataFrame with positive parallax and phot_g_mean_mag.

    Returns:
        New DataFrame with distance_pc and abs_g_mag columns added.
    """
    missing_columns = [col for col in ["parallax", "phot_g_mean_mag"] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns required for absolute magnitude: {missing_columns}")

    if (df["parallax"] <= 0).any():
        raise ValueError("Absolute magnitude requires positive parallax values.")

    df = df.copy()
    df["distance_pc"] = 1000 / df["parallax"]
    df["abs_g_mag"] = df["phot_g_mean_mag"] + 5 * np.log10(df["parallax"]) - 10
    return df

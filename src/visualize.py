"""
Reusable plotting functions for Gaia stellar data.

All functions take a cleaned DataFrame (output of data_clean.add_features())
and return nothing — they display the plot inline in Jupyter.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_parallax_hist(df: pd.DataFrame) -> None:
    """
    Plot the distribution of parallax values.

    Parallax is the primary Gaia measurement. This plot helps assess the
    quality of the sample — most stars should cluster at small parallax
    values (distant stars), with a tail of nearby bright stars at high values.
    """
    plt.figure(figsize=(8, 4))
    df["parallax"].hist(bins=100, color="steelblue", edgecolor="none")
    plt.xlabel("Parallax (mas)")
    plt.ylabel("Number of Stars")
    plt.title("Distribution of Stellar Parallaxes")
    plt.tight_layout()
    plt.show()


def plot_hr_density(df: pd.DataFrame) -> None:
    """
    Plot a density HR diagram using a hex-bin map.

    Uses log-scaled color to reveal structure across 3 orders of magnitude
    in star count. The inverted y-axis is the astronomy convention:
    brighter stars (lower magnitude number) appear at the top.

    Regions to look for:
        Main sequence — diagonal band from top-left (hot blue) to bottom-right (cool red)
        Red giant branch — upper right (bright, red)
        White dwarf sequence — lower left (dim, hot/blue)
    """
    plt.figure(figsize=(8, 8))
    plt.hexbin(df["bp_rp"], df["absolute_mag"], gridsize=50, cmap="inferno", bins="log")
    plt.colorbar(label="log(count)")
    plt.gca().invert_yaxis()  # Astronomy convention: bright (low mag) at top
    plt.xlabel("BP-RP Color  (blue ← → red)")
    plt.ylabel("Absolute Magnitude  (bright ↑ ↓ dim)")
    plt.title("HR Diagram — Density Map")
    plt.tight_layout()
    plt.show()


def plot_abs_mag_hist(df: pd.DataFrame) -> None:
    """
    Plot the distribution of absolute magnitudes.

    The peak reveals the most common stellar brightness in the sample.
    The shape is affected by Malmquist bias (see plot_distance_vs_mag).
    """
    plt.figure(figsize=(8, 4))
    plt.hist(df["absolute_mag"], bins=80, color="steelblue", edgecolor="none")
    plt.xlabel("Absolute Magnitude  (lower = brighter)")
    plt.ylabel("Number of Stars")
    plt.title("Distribution of Absolute Magnitudes")
    plt.tight_layout()
    plt.show()


def plot_distance_vs_mag(df: pd.DataFrame) -> None:
    """
    Plot distance vs absolute magnitude — reveals Malmquist bias.

    Malmquist bias: at larger distances, only intrinsically bright stars are
    detectable above the survey flux limit. This means the sample is not
    representative of all stars — distant faint stars are missing.
    The plot shows this as a missing lower-right corner (faint + far = invisible).
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(df["distance_pc"], df["absolute_mag"], s=1, alpha=0.3, color="steelblue")
    plt.gca().invert_yaxis()
    plt.xlabel("Distance (parsecs)")
    plt.ylabel("Absolute Magnitude  (lower = brighter)")
    plt.title("Distance vs Brightness — Malmquist Bias")
    plt.tight_layout()
    plt.show()


def plot_parallax_hist_log(df: pd.DataFrame) -> None:
    """
    Plot parallax distribution on a log y-axis.

    The log scale makes rare high-parallax stars (nearby) visible alongside
    the dense bulk of distant stars — on a linear scale the tall distant-star
    bar crushes everything else flat.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(df["parallax"], bins=100, color="steelblue", edgecolor="none")
    plt.yscale("log")
    plt.xlabel("Parallax (mas)")
    plt.ylabel("Number of Stars (log scale)")
    plt.title("Distribution of Stellar Parallaxes")
    plt.tight_layout()
    plt.show()


def plot_hr_scatter(df: pd.DataFrame) -> None:
    """
    Plot the HR diagram as a scatter plot — one dot per star.

    Complements plot_hr_density: scatter shows individual stars and outliers
    clearly, while density (hexbin) handles overlapping points better at scale.
    Use s=1, alpha=0.5 to keep 10k points readable.
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(df["bp_rp"], df["absolute_mag"], s=2, alpha=1, color="steelblue")
    plt.gca().invert_yaxis()
    plt.xlabel("BP-RP Color")
    plt.ylabel("Absolute Magnitude")
    plt.title("Gaia HR Diagram — Scatter")
    plt.tight_layout()
    plt.show()


def plot_distance_hist(df: pd.DataFrame) -> None:
    """
    Plot the distribution of distances in parsecs.

    For a volume-limited sample, star counts should rise as d² (more volume
    at larger distances). Deviations from this reveal the survey's completeness
    limit and Malmquist bias. Useful for nearby-star subsets.
    """
    plt.figure(figsize=(8, 4))
    plt.hist(df["distance_pc"], bins=100, color="steelblue", edgecolor="none")
    plt.xlabel("Distance (pc)")
    plt.ylabel("Number of Stars")
    plt.title("Distribution of Stellar Distances")
    plt.tight_layout()
    plt.show()


def plot_hr_clusters(df: pd.DataFrame, n_clusters: int = 4) -> None:
    """
    Plot HR diagram with K-Means cluster labels colored by group.

    Expects df to have a 'cluster' column (integer labels 0..n_clusters-1)
    added by the k-means step in week2.

    Args:
        df: DataFrame with bp_rp, absolute_mag, and cluster columns.
        n_clusters: Number of clusters (used to generate color/label lists).

    Raises:
        ValueError: if 'cluster' column is missing or n_clusters is out of range.
    """
    if "cluster" not in df.columns:
        raise ValueError("DataFrame is missing a 'cluster' column. Run K-Means before calling this function.")

    # Generate colors from a colormap so any n_clusters value works
    cmap = plt.cm.get_cmap("tab10", n_clusters)
    colors = [cmap(i) for i in range(n_clusters)]

    plt.figure(figsize=(9, 9))
    for i in range(n_clusters):
        mask = df["cluster"] == i
        plt.scatter(
            df.loc[mask, "bp_rp"],
            df.loc[mask, "absolute_mag"],
            s=3,
            alpha=0.6,
            color=colors[i],
            label=f"Cluster {i}",
        )

    plt.gca().invert_yaxis()
    plt.xlabel("BP-RP Color  (blue ← → red)")
    plt.ylabel("Absolute Magnitude  (bright ↑ ↓ dim)")
    plt.title("HR Diagram — K-Means Clusters")
    plt.legend(markerscale=4)
    plt.tight_layout()
    plt.show()

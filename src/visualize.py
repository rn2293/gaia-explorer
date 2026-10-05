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
    plt.scatter(df["bp_rp"], df["absolute_mag"], s=10, alpha=1, color="steelblue")
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


def plot_classification_hr(df, *, values=None, title="Gaia brightness teaching label"):
    """Show labels or probabilities on the HR diagram; magnitude is diagnostic only."""
    import numpy as np
    from matplotlib.colors import ListedColormap

    colors = df["is_bright"] if values is None else np.asarray(values)
    cmap = ListedColormap(["#3b6fb6", "#c63d3d"]) if values is None else "coolwarm"
    fig, ax = plt.subplots(figsize=(8, 6))
    points = ax.scatter(df.bp_rp, df.abs_g_mag, c=colors, cmap=cmap, vmin=0, vmax=1, s=8, alpha=0.6)
    ax.axhline(4, color="black", linestyle="--", label="Teaching label cut: M_G = 4")
    ax.invert_yaxis()
    ax.set(xlabel="BP-RP (mag)", ylabel="Absolute G magnitude (uncorrected for extinction)", title=title)
    if values is None:
        # Label both classes beside the plot; a numeric colorbar is easy to miss.
        ax.scatter([], [], color="#3b6fb6", label="Blue dots: 0 = fainter (M_G ≥ 4)")
        ax.scatter([], [], color="#c63d3d", label="Red dots: 1 = brighter (M_G < 4)")
    ax.legend(loc="upper right")
    if values is not None:
        colorbar = fig.colorbar(points, ax=ax)
        colorbar.set_label("Predicted probability of M_G < 4")
    fig.tight_layout()
    plt.show()


def plot_binary_curves(x, curves, *, xlabel, ylabel, title, threshold=None):
    """Compare named curves, including sigmoid, loss, and optimization histories."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, values in curves.items():
        ax.plot(x, values, label=label)
    if threshold is not None:
        ax.axhline(threshold, color="black", linestyle="--", label=f"Threshold = {threshold:g}")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend()
    fig.tight_layout()
    plt.show()


def plot_color_probability(model, df, *, threshold=0.5, title="Color-only classification"):
    """Separate observed 0/1 labels from predicted probabilities.

    Both use numbers between zero and one, but they mean different things. Two
    panels keep a measured class label from looking like a model probability.
    """
    import numpy as np
    grid = pd.DataFrame({"bp_rp": np.linspace(df.bp_rp.min(), df.bp_rp.max(), 300)})
    fig, (observed_ax, probability_ax) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
    )

    # Top: the answers supplied by the Gaia-derived teaching label.
    observed_ax.scatter(df.bp_rp, df.is_bright, s=9, alpha=0.35, color="steelblue")
    observed_ax.set(ylim=(-0.3, 1.3), title="Actual labels: each dot is one validation star")
    observed_ax.set_yticks([0, 1], labels=["Class 0: M_G ≥ 4", "Class 1: M_G < 4"])
    observed_ax.grid(axis="x", alpha=0.2)

    # Bottom: the model's estimate. Only this panel has a probability scale.
    probability_ax.plot(grid.bp_rp, model.predict_proba(grid)[:, 1],
                        color="darkorange", linewidth=2, label="Model probability of label 1")
    probability_ax.axhline(threshold, color="black", linestyle="--",
                           label=f"Decision threshold = {threshold:.0%}")
    probability_ax.set(xlabel="BP-RP color (larger = redder)",
                       ylabel="Estimated chance of label 1", ylim=(-0.05, 1.05),
                       title="Model estimate from color")
    probability_ax.set_yticks([0, 0.5, 1], labels=["0%", "50%", "100%"])
    probability_ax.legend(loc="upper right")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_binary_evaluation(y, probabilities, *, threshold=0.5):
    """Confusion matrix and precision-recall curve for a specified split."""
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay.from_predictions(y, (np.asarray(probabilities) >= threshold).astype(int),
                                           labels=[0, 1], display_labels=["M_G ≥ 4", "M_G < 4"], ax=axes[0], colorbar=False)
    PrecisionRecallDisplay.from_predictions(y, probabilities, ax=axes[1])
    axes[1].axhline(np.mean(y), linestyle="--", color="gray", label="Positive prevalence")
    axes[1].legend()
    fig.tight_layout()
    plt.show()

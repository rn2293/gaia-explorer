# gaia-explorer

A personal reference repo for exploring the **Gaia DR3 stellar catalog** using Python and Jupyter. Covers data querying, cleaning, feature engineering, visualization, and unsupervised machine learning on real star data.

---

## What's Inside

| Notebook | Description |
|----------|-------------|
| `notebooks/week1_gaia_exploration.ipynb` | Query Gaia DR3, clean the data, compute distances and absolute magnitudes, plot the HR diagram |
| `notebooks/week2_kmeans_clustering.ipynb` | Apply K-Means clustering to group stars on the HR diagram without labels |
| `notebooks/week1_nearby_exploration.ipynb` | Placeholder for nearby star deep-dive |

Reusable Python modules live in `src/` — the notebooks import from these so the code stays clean.

---

## Project Structure

```
gaia-explorer/
├── notebooks/          # Jupyter notebooks (one topic per file)
├── src/                # Reusable Python modules
│   ├── data_fetch.py   # Gaia ADQL queries
│   ├── data_clean.py   # Filtering and feature engineering
│   └── visualize.py    # Plotting functions
├── data/
│   ├── raw/            # Raw query results (not committed)
│   ├── processed/      # Cleaned DataFrames (not committed)
│   └── exports/        # CSVs or figures for sharing
├── figures/            # Saved plot outputs
├── requirements.txt
└── .gitignore
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter Lab
jupyter lab
```

Then open any notebook under `notebooks/`.

---

## Data Source

All star data comes from the **Gaia Data Release 3 (DR3)** catalog, queried live via the [Gaia Archive](https://gea.esac.esa.int/archive/) using ADQL through `astroquery.gaia`.

Key table: `gaiadr3.gaia_source`

Key columns used:
- `parallax` — angular displacement in milliarcseconds; used to compute distance (`d = 1000/parallax` in parsecs)
- `phot_g_mean_mag` — apparent magnitude in the G band
- `bp_rp` — color index (blue minus red); proxy for stellar temperature
- `absolute_mag` — derived: `G - 5*log10(d/10)` (the true intrinsic brightness)

---

## Key Concepts

**Hertzsprung-Russell (HR) Diagram** — a scatter plot of stellar color (BP-RP) vs. absolute magnitude. Stars cluster into distinct sequences (main sequence, red giants, white dwarfs) based on their evolutionary stage.

**Parallax-to-distance** — Gaia measures the tiny apparent shift of a star across 6 months of Earth's orbit. `distance_pc = 1000 / parallax_mas`. Only reliable when the signal-to-noise ratio (parallax/parallax_error) is high.

**K-Means clustering** — unsupervised algorithm that groups stars by similarity in color and brightness, recovering the known HR diagram sequences without any labels.

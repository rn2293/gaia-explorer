# gaia-explorer

A personal reference repo for exploring the **Gaia DR3 stellar catalog** using Python and Jupyter. Covers data querying, cleaning, feature engineering, visualization, regression, classification, and clustering on real star data.

---

## What's Inside

| Notebook | Description |
|----------|-------------|
| `notebooks/week1_gaia_exploration.ipynb` | Query Gaia DR3, clean the data, compute distances and absolute magnitudes, plot the HR diagram |
| `notebooks/week2_kmeans_clustering.ipynb` | Apply K-Means clustering to group stars on the HR diagram without labels |
| `notebooks/week1_nearby_exploration.ipynb` | Placeholder for nearby star deep-dive |

Reusable Python modules live in `src/` — the notebooks import from these so the code stays clean.

### Week 3: classification with Gaia

**Start here:** [Week 3 prerequisites](notebooks/week3_prerequisites.ipynb) explains how to choose the Python kernel, restart it, run all cells, and check the Gaia cache.

Follow the same seven-day format as W2. The learning question is: **can stellar color predict whether a source has uncorrected absolute G magnitude below 4?** This is a constructed brightness label, not a verified stellar-type label. The only model input is `bp_rp`; quantities defining the target stay out of the predictors.

| Day | Notebook | W3 coursework and outputs |
|-----|----------|---------------------------|
| 1 | [Classification target](notebooks/week3_day_01_classification_target.ipynb) | Lab 01: cleaning audit, brightness labels, class balance, HR diagram |
| 2 | [Sigmoid probabilities](notebooks/week3_day_02_sigmoid_probabilities.ipynb) | Lab 02: scores, sigmoid, worked probability predictions |
| 3 | [Decision boundaries](notebooks/week3_day_03_decision_boundaries.ipynb) | Lab 03: color boundary, threshold, precision/recall trade-offs |
| 4 | [Logistic loss](notebooks/week3_day_04_logistic_loss.ipynb) | Labs 04–05: cross-entropy, stable cost, regularization term |
| 5 | [Gradient descent](notebooks/week3_day_05_logistic_gradient_descent.ipynb) | Labs 06–07: NumPy training, learning rates, sklearn comparison |
| 6 | [Evaluation and regularization](notebooks/week3_day_06_evaluation_regularization.ipynb) | Labs 07–09: polynomial features, L2, explicit model comparison |
| 7 | [Astronomy reflection](notebooks/week3_day_07_astronomy_reflection.ipynb) | Final held-out evaluation, error analysis, exported results |

The source coursework lives at `../ML-coursework/W3/Optional Labs/`. Each day includes concepts, executable tasks, expected outputs, experiments, and reflection questions with starter answers.

**First run:** W3 reuses `data/processed/gaia_clean_day1.csv` when present; otherwise it fetches and caches a real Gaia sample using the shared W2 loader (internet required). You can also run the W2 [Day 1 cleaning notebook](notebooks/day_01_data_cleaning.ipynb), or set `DATA_PATH` to your own Gaia CSV (`~/` paths are supported). Required columns: `source_id`, `bp_rp`, `phot_g_mean_mag`, `parallax`, `parallax_error`. W3 works from either the repository root or `notebooks/`. Subsequent runs use the cache without querying Gaia. An explicit missing CSV path raises an error. For offline-only use, call `data.load_week3_data(DATA_PATH, use_query_if_missing=False)`. Archive queries verify TLS certificates by default; no synthetic data is substituted. The cleaned dataset needs at least 10 sources in each brightness class.

W3 now follows the W2 learning style: each calculation is visible in small notebook cells. Day 1 computes absolute magnitude, creates the label, makes a stratified 60/20/20 split (seed 42), and saves `data/processed/week3/learning_data.csv`. Run it before Days 2–7. Labels, splits, sigmoid, logistic cost, gradient descent, fitting, and evaluation are all written out in the notebooks. `src/classification.py` only loads and cleans measurements; `src/visualize.py` contains plotting helpers.

Day 2 builds predictions with hand-picked weights. Day 3 trains a baseline and explores thresholds. Day 4 calculates loss and cost. Day 5 defines the gradient and training loop, then compares with scikit-learn. Day 6 visibly compares nine degree/C combinations and saves its choice by validation log loss. The final probability threshold is fixed at 0.5. Day 7 reads the saved choice, retrains on the same training rows, and evaluates the reserved test set. Run Day 6 before Day 7; if Day 1 data changes, rerun Day 6 too. The settings record a dataset fingerprint to detect stale choices.

Outputs under `data/processed/week3/` are `learning_data.csv`, `split_manifest.csv`, `validation_model_comparison.csv`, `experiment_settings.csv`, `test_predictions.csv`, and `test_metrics.csv`. Rerunning the corresponding day replaces its files. The notebooks include accuracy, precision, recall, F1, logistic loss, confusion matrices, precision-recall plots, and constant-baseline comparisons.

Interpret results within this sample: the magnitude cut is an educational choice; dust and parallax uncertainty affect it, and color alone cannot separate all stellar populations. See [ESA's extinction guidance](https://www.cosmos.esa.int/web/gaia/edr3-extinction-law). An unordered archive TOP-N sample is not guaranteed representative of the sky.

Validate from the repository root:

```bash
venv/bin/python -m unittest discover -s tests -v
```

Tests check numerical gradients, extreme scores, sklearn agreement, filtering, split isolation, and the prerequisites plus all seven daily notebooks in temporary directories using an explicitly synthetic fixture. These execution checks are not scientific validation on real Gaia data.


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

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

The `venv/` at the project root has Python 3.14 with all dependencies installed.

## Architecture

This is a personal reference repo for Gaia DR3 stellar data analysis. All reusable logic lives in `src/`; notebooks are thin consumers that call `src/` functions.

### Data pipeline (always in this order)

```python
from src.data_fetch import fetch_gaia_sample   # 1. Query Gaia archive → raw DataFrame
from src.data_clean import clean_sample        # 2. Filter bad measurements → clean DataFrame
from src.data_clean import add_features        # 3. Add distance_pc, absolute_mag
from src import visualize                      # 4. Plot via named functions
```

### `src/` modules

- **`data_fetch.py`** — `fetch_gaia_sample(top_n)`: ADQL query against `gaiadr3.gaia_source`. Calls `_configure_network()` internally which disables SSL verification (corporate network workaround — process-global, irreversible for the session).
- **`data_clean.py`** — `clean_sample(df)`: four-filter quality cut (parallax > 0, parallax_error > 0, SNR > 5, bp_rp not null). `add_features(df)`: adds `distance_pc = 1000/parallax` and `absolute_mag` via the distance modulus formula. Both return copies — inputs are never mutated.
- **`visualize.py`** — one function per plot type. All accept a cleaned DataFrame and call `plt.show()`. `plot_hr_clusters()` requires a `cluster` column and raises `ValueError` if missing.

### Notebooks

Notebooks import from `src/` via `sys.path.insert(0, os.path.abspath('..'))` at the top of each imports cell (notebooks run from `notebooks/`, so `..` resolves to the project root).

- `week1_gaia_exploration.ipynb` — full pipeline: query → explore raw → clean → features → visualize
- `week1_nearby_exploration.ipynb` — same pipeline filtered to `distance_pc < 100 pc`
- `week2_kmeans_clustering.ipynb` — K-Means on HR diagram features; uses `StandardScaler` before fitting; elbow method to choose k

## Key domain facts

- **Parallax → distance**: `distance_pc = 1000 / parallax_mas`. Only reliable when `parallax/parallax_error > 5` (SNR cut). Negative parallax is a noise artifact — never a real distance.
- **Absolute magnitude**: `M = m - 5*log10(d/10)`. The distance modulus corrects apparent brightness for distance. Lower M = intrinsically brighter.
- **HR diagram axes**: x = `bp_rp` (color; low = hot/blue, high = cool/red), y = `absolute_mag` (inverted — bright stars at top by convention).
- **Malmquist bias**: at large distances only intrinsically bright stars are detectable. The clean sample is not a representative census of all stars.

## SSL workaround

`fetch_gaia_sample()` disables SSL verification to reach the Gaia archive through corporate network proxies. To avoid this, set `SSL_CERT_FILE` or `REQUESTS_CA_BUNDLE` to the corporate CA bundle before importing the module.

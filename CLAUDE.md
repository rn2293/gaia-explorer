# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

## Architecture

`src/` modules are the single source of truth — never duplicate their logic inline in notebooks:
- `data_fetch.fetch_gaia_sample(top_n)` — queries Gaia DR3, handles SSL internally
- `data_clean.clean_sample(df)` → `add_features(df)` — filter then compute `distance_pc` and `absolute_mag`
- `visualize.*` — all plots; add new ones here, never write inline matplotlib in notebooks

Every notebook starts with this imports cell:
```python
import sys, os, numpy as np, matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath('..'))
from src.data_fetch import fetch_gaia_sample
from src.data_clean import clean_sample, add_features
from src import visualize
```

Notebooks must be run from the `notebooks/` directory — the `os.path.abspath('..')` depends on it.

## SSL Note

`_configure_network()` in `data_fetch.py` disables SSL verification process-wide (corporate network workaround). Prefer `SSL_CERT_FILE` env var pointing to the corporate CA bundle if possible.

# Real Sample Data

This folder contains a small real-data sample that is already stored in the GitHub repository.

Files:

- `libero_real_sample.npz`
- `libero_real_config.json`

Source:

- built from local `REAL/libero_train`
- image observations were embedded inside parquet rows
- exported with `scripts/build_real_npz.py`

Purpose:

- let the user run a real-data benchmark immediately after `git clone`
- avoid any manual upload step for the first real-data experiment

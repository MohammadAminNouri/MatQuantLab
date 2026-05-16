# V4 Upload Guide

Upload this patch into the same MatQuantLab repository.

Replace existing files when GitHub asks:
- `app/live_dashboard.py`
- `requirements.txt`

Add new files:
- `config/materials_universe.yml`
- `config/global_index_config.yml`
- `config/macro_series_config.yml`
- `config/geopolitical_keywords.yml`
- `data/exposure_score_table.csv`
- `README_V4_READY_TO_COPY.md`

After upload:
1. Open Streamlit Cloud.
2. Open your MatQuantLab app.
3. Click Manage app.
4. Reboot app.
5. The dashboard should show V4 tabs:
   - Executive Radar
   - Prediction Tool for All
   - Material Exposure Matrix
   - Global Stress Map
   - Geopolitical News
   - Regime Detection
   - Validation / Backtest
   - Raw Data

Then copy `README_V4_READY_TO_COPY.md` into your repository `README.md`.

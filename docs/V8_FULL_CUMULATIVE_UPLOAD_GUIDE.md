# MatQuantLab V8 Full Cumulative Upload Guide

Use this package if your GitHub repo is still on V5.

This cumulative patch includes:
- V6/V7 explainable terminal features
- V7 configs: expanded materials, GCIRD/RSS news feeds, geopolitical keywords, macro/global index configs
- V8 ML-only final signal messages
- V8 global manufacturing company universe

Replace these existing files:
- app/live_dashboard.py
- requirements.txt
- config/materials_universe.yml
- config/global_index_config.yml
- config/macro_series_config.yml
- config/geopolitical_keywords.yml
- data/exposure_score_table.csv

Add these new files if missing:
- config/news_sources.yml
- data/manufacturing_company_universe.csv
- README_V7_READY_TO_COPY.md
- README_V8_READY_TO_COPY.md
- docs/V7_UPLOAD_GUIDE.md
- docs/V8_UPLOAD_GUIDE.md
- docs/V8_FULL_CUMULATIVE_UPLOAD_GUIDE.md

After upload:
1. Go to Streamlit Cloud.
2. Manage app.
3. Reboot.
4. Confirm the title says: MatQuantLab V8 — ML Signal Terminal + Global Manufacturing Universe.

Best settings:
- Asset universe: Both
- Interval: 1d
- Lookback: 6mo or 1y
- Model: Ensemble
- Prediction horizon: 5–20
- Shock window: 20

# MatQuantLab V3 Upload Guide

Upload these files into the existing repository and replace old versions when GitHub asks:

- `app/live_dashboard.py`
- `config/experiment_config.yml`
- `docs/README_V3_SECTION_TO_ADD.md`
- `docs/V3_UPLOAD_GUIDE.md`

This patch does not change the existing GitHub Actions pipeline.

After uploading:

1. Go to Streamlit Cloud.
2. Open your MatQuantLab app.
3. Click **Manage app**.
4. Click **Reboot** or **Rerun**.
5. The app should now show:
   - Industrial Stress Radar
   - Lead-Lag Lab
   - ML Explainability
   - AM Watchlist
   - Backtest
   - Raw Data

Then copy the content of `docs/README_V3_SECTION_TO_ADD.md` into the bottom of `README.md`.

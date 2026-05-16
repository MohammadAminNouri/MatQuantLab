# V10 Upload Guide

Replace:
- app/live_dashboard.py
- requirements.txt

Recommended: upload the full zip contents and replace all files when asked.

Then reboot Streamlit Cloud.

If you see intraday period warnings:
- 1m works best with 1d/5d
- 5m/15m works best with 5d/1mo
- 5y/10y requires interval=1d

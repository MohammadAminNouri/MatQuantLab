# MatQuantLab — Clear Decision Cockpit

This version removes vague dashboard language and forces a clear path:

```text
tape → drivers → chain → ML verdict → scenario → news
```

## Key changes

- Replaces confusing "large upward pressure" language with:
  - what moved
  - whether it is unusual
  - effect on the focus instrument
  - why it matters
- Adds a "Read Me First" page with the product logic.
- Scenario lab now has presets and hides meaningless zero-shock charts.
- Transmission chart is not shown for a single instrument because a one-bar chart is not useful.
- Driver explanations are separated into macro, material-chain, own-tape and basket layers.
- ML verdicts are still available but clearly marked as research diagnostics.
- Uses actual prices, volume, market value, alpha/beta and 52-week context.

## Upload

Replace:

- app/live_dashboard.py
- requirements.txt

Add:

- data/global_manufacturing_universe_extended.csv
- README_CLEAR_DECISION_COCKPIT.md
- docs/CLEAR_DECISION_COCKPIT_UPLOAD_GUIDE.md

Then reboot Streamlit Cloud.

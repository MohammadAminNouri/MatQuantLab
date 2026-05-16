# Beginner upload steps

You said you cannot work from your computer. Use this as the simplest path.

## What you have to do

1. Download the zip file I gave you.
2. Extract/unzip it.
3. Open your GitHub repository.
4. Click **Add file** → **Upload files**.
5. Upload everything inside the extracted folder.
6. Choose **Commit directly to main**.
7. Click **Commit changes**.

## After upload

Go to your GitHub repository:

1. Click **Actions**.
2. Click **Run MatQuantLab Research Pipeline**.
3. Click **Run workflow**.
4. Wait until it finishes.

It will run the Python pipeline on GitHub servers, not your computer.

## If GitHub Actions fails

Do not panic. Open the failed action, copy the red error text, and ask ChatGPT to fix it.

## What should appear after the workflow works

```text
outputs/figures/cmsi.png
outputs/figures/signal_decay_heatmap.png
outputs/figures/model_leaderboard.png
outputs/figures/feature_importance.png
outputs/figures/backtest_equity_curve.png
outputs/research_summary.md
```

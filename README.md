MatQuantLab
Critical Materials Intelligence Layer for Cross-Asset ML Research
MatQuantLab is a materials-informed quantitative research project that tests whether signals from metals, energy, USD pressure, volatility, and industrial stress contain lead-lag information for assets linked to:
mining and metals
copper producers
aerospace and defense
industrials
additive manufacturing
broad market benchmarks
Unlike generic “stock prediction” projects, MatQuantLab starts from a materials-engineering hypothesis:
> Physical supply-chain stress in metals, energy, USD pressure, and industrial volatility may contain early information about industrial equity risk.
This repository is built as a portfolio/research project, not as financial advice or a live trading system.
---
Live Dashboard
The interactive dashboard is deployed with Streamlit:
Live app: https://matquantlab-gegdszewybwbbbdvm5kbnc.streamlit.app/
Main Streamlit file:
```text
app/live_dashboard.py
```
The dashboard includes:
Industrial Stress Radar
Critical Materials Stress Index v2
Lead-Lag Signal Lab
ML Explainability
Additive Manufacturing Watchlist
Toy research backtest
Raw price/feature inspection
---
Core Research Question
Can critical-materials and macro stress signals predict future returns or drawdown risk in industrial, mining, aerospace, defense, and additive-manufacturing-linked assets?
MatQuantLab tests whether information from:
copper
aluminum
oil
gold
USD index
VIX
metals-sector ETFs
aerospace/defense ETFs
additive-manufacturing-linked equities
has measurable relationships with future asset returns.
---
Why This Project Is Different
Most beginner quant projects ask:
> Can machine learning predict stock prices?
MatQuantLab asks a more domain-specific question:
> Can materials-market stress and industrial supply-chain information be transformed into interpretable cross-asset signals?
The niche angle is the combination of:
```text
materials engineering
commodity shocks
industrial supply chains
aerospace and defense exposure
additive manufacturing assets
machine learning
lead-lag analysis
interactive dashboarding
```
This makes the project more connected to materials engineering and metallurgical/industrial knowledge than generic finance ML projects.
---
Critical Materials Stress Index v2
The dashboard builds a weighted stress index from commodity and macro proxies:
Component	Interpretation
Copper	industrial demand and electrification proxy
Aluminum	energy-intensive manufacturing proxy
Oil	transport, energy, and production-cost pressure
USD	commodity-pressure and global liquidity proxy
VIX	market stress and risk appetite
Gold	defensive stress / safe-haven proxy
The dashboard classifies the stress state as:
```text
Normal
Elevated stress
Extreme stress
Relief / easing
```
---
Asset Universe
Example assets currently used:
Ticker	Theme
XME	mining and metals producers
COPX	copper miners
PICK	global metals and mining
XLB	US materials sector
ITA	defense and aerospace
XAR	aerospace supply chain
XLI	industrials
VIS	broad industrials
PRNT	additive manufacturing ETF
DDD	additive manufacturing single name
SSYS	additive manufacturing single name
MTLS	AM software/services
NNDM	advanced electronics / AM
SPY	broad market benchmark
---
Dashboard Tabs
1. Industrial Stress Radar
Shows CMSI v2, metals stress, macro-risk stress, current stress drivers, and early-warning notes.
2. Lead-Lag Lab
Tests whether stress features have a relationship with future returns at the selected horizon.
Outputs include:
Spearman information coefficient
Pearson correlation
number of aligned observations
3. ML Explainability
Trains a simple research model on the selected asset and horizon.
Current model options:
Ridge Regression
Random Forest
The tab shows:
RMSE
prediction correlation
prediction vs realized future return
feature importance
latest model-implied pressure
4. Additive Manufacturing Watchlist
Tracks AM-linked assets and tests their relationship to the Critical Materials Stress Index.
Example AM watchlist:
```text
PRNT
DDD
SSYS
MTLS
NNDM
```
5. Backtest
A toy research backtest:
```text
long when prediction > 0
short when prediction < 0
```
This is only for research visualization. It does not include professional transaction-cost, slippage, borrow-cost, liquidity, or capacity modeling.
6. Raw Data
Displays downloaded prices and created features for transparency.
---
Data Source
The live dashboard uses Yahoo Finance data through `yfinance`.
Example proxies:
Signal	Yahoo ticker
Copper futures	HG=F
Aluminum futures	ALI=F
Oil futures	CL=F
Gold futures	GC=F
USD index	DX-Y.NYB
VIX	^VIX
Important limitations:
free data can be delayed
intraday data can be incomplete or rate-limited
Yahoo/yfinance is not institutional-grade market data
results may change depending on data availability
For professional use, the data layer should be replaced with audited market data.
---
Automated Research Pipeline
The repository also includes a GitHub Actions pipeline that generates research outputs automatically.
Generated outputs include:
```text
outputs/research_summary.md
outputs/figures/cmsi.png
outputs/figures/signal_decay_heatmap.png
outputs/figures/model_leaderboard.png
outputs/figures/backtest_equity_curve.png
outputs/figures/feature_importance.png
data/processed/
```
Run it manually from:
```text
Actions → Run MatQuantLab Research Pipeline → Run workflow
```
---
Automated Research Outputs
Critical Materials Stress Index
![CMSI](outputs/figures/cmsi.png)
Lead-Lag Signal Heatmap
![Heatmap](outputs/figures/signal_decay_heatmap.png)
ML Model Comparison
![Models](outputs/figures/model_leaderboard.png)
Strategy Equity Curve
![Backtest](outputs/figures/backtest_equity_curve.png)
Feature Importance
![Importance](outputs/figures/feature_importance.png)
---
How to Run Locally
Clone the repository:
```bash
git clone https://github.com/MohammadAminNouri/MatQuantLab.git
cd MatQuantLab
```
Install dependencies:
```bash
pip install -r requirements.txt
```
Run the live dashboard:
```bash
streamlit run app/live_dashboard.py
```
Run the automated research pipeline if available:
```bash
python scripts/run_daily_update.py
```
---
How to Use the Live App
Suggested first experiment:
```text
Target asset: XME
Interval: 5m
Period: 5d
Model: Ridge
Prediction horizon: 12 bars
Shock window: 20 bars
```
Then test:
```text
COPX
ITA
XAR
PRNT
DDD
SSYS
MTLS
```
Look at:
CMSI v2 value
stress regime
top stress drivers
lead-lag IC table
feature importance
latest model-implied pressure
---
Example Research Questions
MatQuantLab can be used to explore questions such as:
Do copper shocks lead copper-mining equities?
Does USD strength weaken metals-linked assets?
Does oil stress affect aerospace and industrial names?
Does CMSI v2 predict drawdown risk in mining ETFs?
Are additive-manufacturing-linked equities more sensitive to industrial stress than broad industrial ETFs?
Do ML models add information beyond simple lead-lag correlations?
Which stress features matter most for each asset group?
---
Current Status
This project is currently a research/portfolio prototype.
Working components:
Streamlit interactive dashboard
Yahoo/yfinance data download
CMSI v2 stress index
lead-lag correlation analysis
Ridge and Random Forest modeling
feature importance
AM watchlist
toy backtest
GitHub Actions research output generator
Still to improve:
proper transaction-cost modeling
walk-forward validation
factor neutralization
sector-neutral portfolio construction
survivorship-bias controls
professional market data source
model stability analysis
regime clustering
better risk metrics
---
Roadmap
Planned upgrades:
sector-neutral ranking
rolling Sharpe and drawdown metrics
transaction-cost and turnover model
walk-forward validation by date
regime clustering
feature orthogonalization
factor exposure decomposition
Streamlit downloadable reports
better additive manufacturing exposure map
industrial supply-chain news/sentiment layer
---
Research Disclaimer
This repository is for educational and research purposes only.
It is not:
investment advice
a trading recommendation
a production trading system
a guaranteed alpha model
Free market data may contain errors, missing values, delays, and survivorship bias. Any financial interpretation should be treated as experimental.
---
One-Line Summary
MatQuantLab converts critical-materials and macro stress information into interpretable cross-asset ML signals for mining, aerospace, defense, industrial, and additive-manufacturing-linked assets.

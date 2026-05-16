# MatQuantLab research plan

## Core question

Can materials-engineering knowledge improve financial prediction in metals, mining, aerospace, defense, industrial, and additive-manufacturing equities?

## Why this is rare

The rare part is not merely using machine learning. The rare part is connecting physical industrial inputs, supply-chain stress, commodity markets, energy costs, and equity returns.

## Research modules

### 1. Critical Materials Stress Index

Build a rolling stress index from commodity and macro shocks.

### 2. Signal decay

Test whether each feature predicts 5D, 20D, or 60D future returns.

### 3. ML model comparison

Compare Ridge, ElasticNet, Random Forest, and Gradient Boosting using walk-forward validation.

### 4. Transaction-cost backtest

Rank assets by predicted return, go long the top group, short the bottom group, and subtract trading costs.

### 5. Regime detection

Cluster market regimes and test whether the signal works only in specific environments.

### 6. Overfitting diagnostics

Use permutation tests and basic IC warnings to avoid fake discoveries.

## First publishable result

Does CMSI predict 20-day future returns of XME, COPX, PICK, XAR, ITA, XLI, VIS, and PRNT?

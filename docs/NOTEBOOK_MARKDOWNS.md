# Markdown text for notebooks

## 01 Data Collection

This notebook builds the raw data layer for MatQuantLab. The goal is to collect updatable prices for commodities, macro proxies, sector ETFs, and advanced-manufacturing-linked equities.

The key rule is timestamp discipline: a feature must be known at or before the prediction date.

## 02 Feature Engineering

This notebook converts raw price series into commodity shocks, macro shocks, momentum, volatility, drawdown, and the Critical Materials Stress Index.

## 03 Signal Decay Analysis

A signal is not simply good or bad. It has a half-life. This notebook tests whether each feature is more useful for 5D, 20D, or 60D future returns.

## 04 ML Prediction Models

This notebook trains machine-learning models using walk-forward validation. Random shuffling is not allowed because it leaks time structure.

## 05 Backtest With Costs

A prediction is not economically useful unless it survives transaction costs. This notebook compares gross and net performance.

## 06 Regime Detection

This notebook tests whether the model works only during certain macro/commodity regimes.

## 07 Final Report

The final notebook combines the results and asks whether the signal is real, tradable, robust, explainable, and unique.

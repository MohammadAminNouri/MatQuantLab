# MatQuantLab

Material-driven industrial risk research.

MatQuantLab traces how material, energy, macro and event shocks transmit into selected assets.

Core idea:

```text
material shock → industrial exposure → market response
```

## What is different in this version

- Everything changes around the assets selected by the user.
- Charts, event terms, transmission map and model drivers are selection-specific.
- Explanations are attached to the relevant chart/table.
- Futures, commodities, stocks and options are separated.
- Forecast horizon is stated as a research-validity window.
- AM and powder metallurgy are treated as a real materials/process chain:
  - titanium, nickel, aluminum and powder feedstock
  - LPBF, EBM, HIP and metal powder terms
  - process energy and natural gas
  - aerospace/defense demand
  - powder metallurgy and alloy input pressure

## Recommended settings

Longer-horizon research:

```text
Data interval: 1d
History: 5y or 10y
Model selection: Adaptive
Forecast horizon: 5–20 bars
Shock window: 20
```

Fast monitoring:

```text
Data interval: 5m or 15m
History: 5d or 1mo
Forecast horizon: 3–12 bars
Fast mode: on
```

Signals are research diagnostics, not investment advice.

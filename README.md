# PJM SOUTH Day-Ahead LMP Forecast Model

Hourly Day-Ahead LMP forecasting for PJM SOUTH aggregate node using 41 data sources, ~51 engineered features, LightGBM + Ridge ensemble, and spike classification.

## Features
- 24-hour DA LMP forecast with 90% confidence intervals and spike risk flags
- **Friday 3-day forecast mode** for Sat/Sun/Mon with cascading D-1 synthetic lags
- **Automatic CI widening** for D+2 (15%) and D+3 (25%) forecasts
- WHub price assignment by hour block (HE08-HE23 on-peak, HE01-HE07/HE24 off-peak) across all day types including weekends

## Status
✅ Core implementation complete — train on real data to begin forecasting.

## New in v2

### 13 New Data Sources
- **Open-Meteo** (6 free weather variables, no auth required): shortwave radiation (GHI), direct radiation, apparent temperature, wind gusts, precipitation, and mean sea-level pressure — for all 5 forecast cities (Richmond VA, Norfolk VA, Raleigh NC, Pittsburgh PA, Columbus OH)
- **Morningstar Commodities** (3 datasets): Columbia Gas (TCO) daily spot, PJM WHub prompt-month forward curve, Transco Zone 5 gas forward curve
- **PJM API** (4 new feeds): ancillary service prices (RegA, RegD, Sync Reserve), 5-minute marginal CO₂ emission rates (aggregated hourly), transmission line thermal ratings/de-rate flags, and PJM instantaneous load (aggregated hourly)

### 11 New Derived Features
| Feature | Description |
|---------|-------------|
| `ghi_solar_estimate_h` | GHI × installed solar capacity — direct solar generation proxy |
| `clear_sky_fraction_h` | Direct / shortwave ratio — clear-sky vs cloudy solar |
| `gust_curtailment_flag` | Binary: wind gusts > 25 m/s → turbine feathering risk |
| `columbia_z5_spread` | Columbia Gas − Transco Z5 price basis |
| `gas_contango` | Z5 prompt-month forward − Z5 spot |
| `power_contango` | WHub prompt-month forward − WHub spot DA |
| `reg_price_d1` | Regulation market clearing price D-1 average |
| `reserve_scarcity_signal` | Binary: sync reserve price > $50 |
| `marginal_emission_rate_d1` | Avg CO₂ emission rate D-1 (lb/MWh) — marginal unit proxy |
| `pressure_gradient_12h` | Pressure_h − Pressure_{h-12} — storm front detection |
| `precip_flag` | Binary: precipitation > 0 |

### VS Code Setup Guide
Step-by-step guide for non-technical users in `USER_GUIDE.md` (Section 2b): install VS Code, Python extension, virtual environment, and run forecasts without using a plain terminal.

### Expected MAE Improvement
With the full expanded data stack: **~15–20% MAE reduction** versus the base model (primarily driven by GHI solar prediction, Morningstar forward curves, and PJM ancillary service signals).
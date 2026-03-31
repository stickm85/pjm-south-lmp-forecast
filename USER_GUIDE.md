# PJM SOUTH DA LMP Forecast — User Guide

---

## Section 1: What This Tool Does

This tool predicts **next-day hourly electricity prices** for the **PJM SOUTH aggregate pricing node**. Every morning before 09:00 Eastern Prevailing Time (EPT), you provide three market inputs and the tool produces 24 hourly price forecasts for the following day — one for each Hour Ending (HE01 through HE24).

### What You Get

| Output | Meaning |
|--------|---------|
| **24 Hourly Forecasts** | Predicted Day-Ahead LMP ($/MWh) for HE01–HE24 EPT |
| **90% Confidence Interval** | Lower and upper price bounds you can expect to be right ~90% of the time |
| **Spike Risk** | Low / Moderate / High / Very High — likelihood of a price spike above $100/MWh |
| **On-Peak / Off-Peak Label** | Whether each hour is on-peak per PJM market rules |

### Your Three Inputs

You need three numbers, all readily available by 09:00 EPT:

1. **Western Hub DA On-Peak Price ($/MWh)** — The market consensus for next-day PJM Western Hub on-peak electricity. "On-Peak" = Hours Ending 07 through 23, Monday through Friday, excluding NERC holidays.

2. **Western Hub DA Off-Peak Price ($/MWh)** — Same node, but for off-peak hours (all other hours not classified above).

3. **Transco Zone 6 NNY Gas Price ($/MMBtu)** — Next-day natural gas price at Transco Zone 6 Non-NY delivery point. This is the marginal fuel price that strongly drives SOUTH electricity prices.

---

## Section 2: One-Time Setup

### Installing Python (Windows)

1. Open your browser and go to **https://www.python.org/downloads/windows/**
2. Download the latest Python 3.9+ installer (e.g., `python-3.12.x-amd64.exe`)
3. Run the installer — **IMPORTANT: check the box that says "Add Python to PATH"** before clicking Install
4. After installation, open **Command Prompt** (press `Win + R`, type `cmd`, press Enter)
5. Verify by typing: `python --version` — you should see `Python 3.12.x` or similar

### Installing Python (Mac)

1. Open Terminal (press `Cmd + Space`, type `Terminal`, press Enter)
2. Install Homebrew if you don't have it: paste this command and press Enter:
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. Then install Python:
   ```
   brew install python@3.12
   ```
4. Verify: `python3 --version`

### Downloading the Project

**Option A — GitHub ZIP (no Git required):**
1. Go to the project GitHub page
2. Click the green **Code** button → **Download ZIP**
3. Extract the ZIP to a folder, e.g., `C:\Users\YourName\pjm-south-lmp-forecast\`

**Option B — Git clone:**
```bash
git clone https://github.com/your-org/pjm-south-lmp-forecast.git
cd pjm-south-lmp-forecast
```

### Installing Dependencies

Open a terminal/Command Prompt and navigate to the project folder:

```bash
# Windows
cd C:\Users\YourName\pjm-south-lmp-forecast

# Mac/Linux
cd ~/pjm-south-lmp-forecast
```

Then run:
```bash
pip install -r requirements.txt
```

This installs all required libraries (LightGBM, scikit-learn, pandas, etc.). This takes 2–5 minutes on first run.

### Configuring API Keys (Optional)

The tool works out-of-the-box with **mock data** for testing. To use real market data:

1. Open the file `config/settings.yaml` in any text editor (Notepad, TextEdit, VS Code)
2. Fill in your API credentials:

```yaml
pjm:
  api_key: "YOUR_PJM_DATAMINER_API_KEY"  # From https://api.pjm.com

gas:
  api_key: "YOUR_GAS_API_KEY"  # ICE, Platts, or NGI credentials
```

> **Note:** The tool runs fully with mock data when no API keys are set. Mock data is useful for testing and training but should not be used for actual trading decisions.

---

## Section 3: Daily Usage (09:00 EPT)

### Getting Your Input Values

**Western Hub On-Peak & Off-Peak Prices:**
- Check your ICE screen or broker sheet for "PJM Western Hub" Day-Ahead
- Look for "On-Peak" (also written as "5×16" or "peak") and "Off-Peak" (or "7×8" and weekends) prices
- Typical On-Peak range: $30–$80/MWh; Off-Peak: $20–$60/MWh

**Transco Zone 6 NNY Gas Price:**
- Check ICE Natural Gas screen for "Transco Zone 6 Non-NY" next-day gas
- Or call your gas broker for the daily index price
- Typical range: $2.50–$8.00/MMBtu (higher in winter)

### Running the Forecast

Open a terminal in the project folder and run:

```bash
python cli.py forecast --whub-onpeak 45.50 --whub-offpeak 28.75 --gas 3.42
```

**Breaking down the command:**
- `python cli.py` — runs the program
- `forecast` — tells it to generate a forecast (not train or evaluate)
- `--whub-onpeak 45.50` — your WHub on-peak price input ($45.50/MWh)
- `--whub-offpeak 28.75` — your WHub off-peak price input ($28.75/MWh)
- `--gas 3.42` — your gas price input ($3.42/MMBtu)

**Optional flags:**
```bash
# Forecast for a specific date (instead of tomorrow)
python cli.py forecast --whub-onpeak 45.50 --whub-offpeak 28.75 --gas 3.42 --date 2026-01-15

# Save output as CSV
python cli.py forecast --whub-onpeak 45.50 --whub-offpeak 28.75 --gas 3.42 --output csv --output-file forecast.csv

# Save as JSON
python cli.py forecast --whub-onpeak 45.50 --whub-offpeak 28.75 --gas 3.42 --output json
```

### Training the Model (First Time)

Before the model can produce ML-based forecasts, you need to train it once:

```bash
python cli.py train --start-date 2023-01-01 --end-date 2025-12-31
```

This takes approximately 5–15 minutes and saves the model to `models/ensemble.pkl`. After training, forecasts use the full ML ensemble.

### Scheduling (Optional)

**Windows — Task Scheduler:**
1. Open Task Scheduler (search in Start menu)
2. Click **Create Basic Task** → name it "PJM SOUTH Forecast"
3. Set trigger: **Daily**, at **9:00 AM**
4. Action: **Start a program**
   - Program: `python`
   - Arguments: `cli.py forecast --whub-onpeak 45.50 --whub-offpeak 28.75 --gas 3.42 --output csv --output-file C:\forecasts\today.csv`
   - Start in: `C:\Users\YourName\pjm-south-lmp-forecast`

**Mac/Linux — cron job:**
```bash
# Open crontab editor
crontab -e

# Add this line (runs at 9:00 AM Eastern — adjust timezone as needed)
0 9 * * 1-5 cd /home/yourname/pjm-south-lmp-forecast && python cli.py forecast --whub-onpeak 45.50 --whub-offpeak 28.75 --gas 3.42 --output csv --output-file ~/forecasts/today.csv
```

> **Note:** You'll need to update the price inputs each morning before the scheduled run, or integrate with an automated price data feed.

---

## Section 4: Reading the Output

### Sample Output

```
============================================================
  PJM SOUTH DA LMP FORECAST
  Target Date: Monday, January 15, 2026
============================================================
  Inputs:
    WHub On-Peak:  $45.00/MWh
    WHub Off-Peak: $28.75/MWh
    Gas (Z6 NNY):  $3.420/MMBtu
============================================================

Hour EPT    Forecast   Lower 90%   Upper 90%   WHub DA    Period       Spike Risk
--------  ----------  ----------  ----------  ---------  -----------  ----------
HE01         $30.75      $26.14      $35.36      $28.75  Off-Peak     Low
HE02         $29.50      $25.08      $33.93      $28.75  Off-Peak     Low
...
HE08         $48.50      $41.23      $55.78      $45.00  On-Peak      Low
HE09         $52.25      $44.41      $60.09      $45.00  On-Peak      Moderate
...
HE16         $65.40      $55.59      $75.21      $45.00  On-Peak      Moderate
```

### Column Explanations

| Column | Description |
|--------|-------------|
| **Hour EPT** | Hour Ending in Eastern Prevailing Time. HE01 = midnight-to-1am, HE08 = 7am-to-8am, HE24 = 11pm-to-midnight |
| **Forecast** | The model's best estimate of the PJM SOUTH DA LMP for that hour ($/MWh) |
| **Lower 90%** | The lower bound of the 90% confidence interval — actual prices should be above this ~95% of the time |
| **Upper 90%** | The upper bound — actual prices should be below this ~95% of the time |
| **WHub DA** | The Western Hub price you provided (on-peak or off-peak depending on the hour) |
| **Period** | Whether the hour is "On-Peak" or "Off-Peak" per PJM market rules |
| **Spike Risk** | Probability category for a price spike >$100/MWh (see below) |

### Understanding 90% Confidence Intervals

The **Lower 90%** and **Upper 90%** values define a range where the actual day-ahead price should fall about **90% of the time** based on historical patterns.

**Think of it this way:** If the forecast shows:
- Forecast: $52.25/MWh
- Lower 90%: $44.41/MWh  
- Upper 90%: $60.09/MWh

This means: "We expect the actual DA price to land between $44.41 and $60.09 about 9 times out of 10."

**Wide intervals** (e.g., $30–$80) indicate high uncertainty — common during extreme weather, grid stress events, or unusual fuel prices.

**Narrow intervals** (e.g., $45–$55) indicate higher confidence — typical for moderate weather days.

### Understanding Spike Risk

| Label | Probability | Meaning | Suggested Action |
|-------|-------------|---------|-----------------|
| **Low** | < 10% | Normal market conditions | Standard operations |
| **Moderate** | 10–35% | Elevated risk, watch conditions | Alert trading desk, check reserve margin |
| **High** | 35–65% | Significant spike probability | Reduce long exposure, review open position |
| **Very High** | > 65% | Near-certain spike event | Emergency protocols, contact operations |

Spike risk is driven by: high gas prices relative to hub prices, low reserve margin, prior-day EEA events, and extreme weather conditions.

---

## Section 5: Troubleshooting FAQ

### "python is not recognized as an internal or external command"
**Windows only.** This means Python wasn't added to your PATH during installation.

**Fix:** Re-run the Python installer and check "Add Python to PATH", or manually add Python to your system PATH:
1. Search "Environment Variables" in Start menu
2. Edit `PATH` → Add `C:\Python312\` and `C:\Python312\Scripts\`

### "ModuleNotFoundError: No module named 'lightgbm'" (or similar)
You need to install the dependencies.

**Fix:**
```bash
pip install -r requirements.txt
```
If that fails, try:
```bash
python -m pip install -r requirements.txt
```

### "No trained model found" warning
The model hasn't been trained yet. The tool will use a simpler rule-based forecast instead.

**Fix:** Train the model once:
```bash
python cli.py train --start-date 2023-01-01 --end-date 2025-12-31
```

### API key errors
If you see `NotImplementedError: Set pjm.api_key in config/settings.yaml`, your API key is not configured.

**Fix options:**
1. Add your API key to `config/settings.yaml` (see Section 2)
2. Leave the key blank — the tool will automatically use mock data (appropriate for testing only)

### Very slow first run
The first forecast may take 30–60 seconds as Python loads all libraries. Subsequent runs are faster.

### "PermissionError" when saving files
The tool doesn't have write permission to the output directory.

**Fix:** Run from the project directory, or specify a different output path:
```bash
python cli.py forecast --whub-onpeak 45.0 --whub-offpeak 30.0 --gas 3.5 --output-file ~/Desktop/forecast.csv
```

---

## Section 6: Quick Reference Card

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PJM SOUTH DA LMP FORECAST — QUICK REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DAILY COMMAND (09:00 EPT):
  python cli.py forecast \
    --whub-onpeak  [PRICE]  \   ← WHub On-Peak $/MWh
    --whub-offpeak [PRICE]  \   ← WHub Off-Peak $/MWh
    --gas          [PRICE]      ← Transco Z6 NNY $/MMBtu

EXAMPLE:
  python cli.py forecast --whub-onpeak 45.50 --whub-offpeak 28.75 --gas 3.42

ALL COMMANDS:
  python cli.py forecast  --help    Generate price forecast
  python cli.py train     --help    Train ML model on historical data
  python cli.py evaluate  --help    Run backtester, show performance
  python cli.py ingest    --help    Fetch & cache latest data
  python cli.py tune      --help    Optimize hyperparameters

OUTPUT FORMATS:
  (default)        Pretty table to terminal
  --output csv     CSV format
  --output json    JSON format
  --output-file f  Save to file f instead of printing

ON-PEAK DEFINITION:
  HE07–HE23, Monday–Friday, excluding NERC holidays

NERC HOLIDAYS (2025):
  Jan 1, May 26, Jul 4, Sep 1, Nov 27, Dec 25

SPIKE RISK THRESHOLDS:
  Low       < 10%  │  Moderate  10–35%
  High    35–65%   │  Very High  > 65%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

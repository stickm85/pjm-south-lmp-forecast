# PJM SOUTH DA LMP Forecast — User Guide (Windows / VS Code)

---

## Section 1: What This Tool Does

This tool predicts **next-day hourly electricity prices** for the **PJM SOUTH aggregate pricing node**. Every morning before 09:00 Eastern Prevailing Time (EPT), you provide three market inputs and the tool returns a full 24-hour price forecast with confidence intervals and spike risk flags.

### What You Get

| Output | Meaning |
|--------|---------|
| **24 Hourly Forecasts** | Predicted Day-Ahead LMP ($/MWh) for HE01–HE24 EPT |
| **90% Confidence Interval** | Lower and upper price bounds you can expect to be right ~90% of the time |
| **Spike Risk** | Low / Moderate / High / Very High — likelihood of a price spike above $100/MWh |
| **On-Peak / Off-Peak Label** | Whether each hour is on-peak per PJM market rules |

### Your Three Inputs

You need three numbers, all readily available by 09:00 EPT:

1. **Western Hub DA On-Peak Price ($/MWh)** — The market consensus for next-day PJM Western Hub on-peak electricity. "On-Peak" = Hours Ending 08 through 23, Monday through Friday, excluding NERC holidays.

2. **Western Hub DA Off-Peak Price ($/MWh)** — Same node, but for off-peak hours (all other hours not classified above).

3. **Transco Zone 5 Gas Price ($/MMBtu)** — Next-day natural gas price at Transco Zone 5 delivery point (Virginia/North Carolina). This is the marginal fuel price that most directly drives SOUTH electricity prices.

---

## Section 2: One-Time Setup

This guide covers setup on **Windows** using **Visual Studio Code (VS Code)**.

### Step 1: Install Python

1. Open your browser and go to **https://www.python.org/downloads/windows/**
2. Download the latest Python 3.13 installer (e.g., `python-3.13.x-amd64.exe`)
3. Run the installer — **IMPORTANT: check the box that says "Add Python to PATH"** before clicking **Install Now**
4. After installation, open **Command Prompt** (press `Win + R`, type `cmd`, press Enter)
5. Verify by typing:
   ```
   python --version
   ```
   You should see `Python 3.13.x` or similar.

> **If `python` is not recognized:** Re-run the Python installer, click **Modify**, and make sure **"Add Python to environment variables"** is checked. Alternatively, manually add Python to your PATH:
> 1. Search **"Environment Variables"** in the Start menu
> 2. Click **"Edit the system environment variables"** → **Environment Variables**
> 3. Under **User variables**, select `Path` → **Edit** → **New**
> 4. Add `C:\Users\YourName\AppData\Local\Programs\Python\Python313\` and `C:\Users\YourName\AppData\Local\Programs\Python\Python313\Scripts\`

### Step 2: Install Visual Studio Code

1. Open your browser and go to **https://code.visualstudio.com/**
2. Click the big blue **Download for Windows** button
3. Run the installer. On the **"Select Additional Tasks"** screen, check **all boxes** — especially:
   - ✅ "Add to PATH (requires shell restart)"
   - ✅ "Register Code as an editor for supported file types"
   - ✅ "Add 'Open with Code' action to Windows Explorer"
4. Open VS Code after installation completes

### Step 3: Install the Python Extension in VS Code

1. In VS Code, click the **square icon** on the left sidebar (or press `Ctrl+Shift+X`) to open the Extensions panel
2. In the search box, type **Python**
3. Click **Install** on the extension by **Microsoft** — it has a blue logo and millions of downloads
4. Wait for the installation to finish (a progress bar appears at the bottom)

### Step 4: Download the Project

**Option A — GitHub ZIP (no Git required):**
1. Go to **https://github.com/stickm85/pjm-south-lmp-forecast**
2. Click the green **Code** button → **Download ZIP**
3. Extract the ZIP to a folder, e.g., `C:\Users\YourName\pjm-south-lmp-forecast\`

**Option B — Git clone (if Git is installed):**
1. Open Command Prompt and run:
   ```
   git clone https://github.com/stickm85/pjm-south-lmp-forecast.git
   ```

### Step 5: Open the Project in VS Code

1. In VS Code, go to **File → Open Folder**
2. Navigate to where you downloaded or extracted `pjm-south-lmp-forecast`
3. Click **"Select Folder"**
4. If VS Code asks **"Do you trust the authors of this folder?"**, click **"Yes, I trust the authors"**
5. You should now see all project files listed in the left sidebar (Explorer panel)

### Step 6: Set Up a Python Virtual Environment

A virtual environment keeps this project's packages separate from your system Python, preventing conflicts.

Open the VS Code built-in terminal:
- Press `` Ctrl+` `` (the backtick key just below Escape), **or**
- Go to **Terminal → New Terminal** in the menu bar

> **Important:** If the terminal prompt starts with `PS C:\...>` (PowerShell) and you have trouble with commands below, switch to Command Prompt by clicking the **dropdown arrow (˅)** next to the **+** icon in the terminal panel and selecting **"Command Prompt"**.

Type these commands **one at a time**, pressing **Enter** after each:

```
python -m venv venv
```
```
venv\Scripts\activate
```

Your terminal prompt should now show `(venv)` at the beginning. If it does, continue:

```
python -m pip install -r requirements.txt
```

> **Why `python -m pip` instead of just `pip`?** Using `python -m pip` ensures pip runs under the correct Python environment. Bare `pip` sometimes points to a different Python installation or is missing from PATH entirely — this is the most common cause of "pip is not recognized" errors.

This installs all required libraries (LightGBM, scikit-learn, pandas, etc.). This takes 2–5 minutes on first run.

**If `python -m pip` still fails:**
```
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
```

### Step 7: Select the Python Interpreter

VS Code needs to know which Python to use (the one in your virtual environment):

1. Press `Ctrl+Shift+P` to open the **Command Palette**
2. Type **Python: Select Interpreter** and click it
3. Choose the option that shows `.\venv\Scripts\python.exe` — this is the virtual environment you just created
4. The Python version now appears in the bottom-left status bar of VS Code

### Step 8: Verify Installation — Run Tests

Confirm everything is installed correctly by running the test suite.

**Option A — VS Code terminal (recommended):**

Make sure `(venv)` appears at the start of your terminal prompt, then run:

```
python -m pytest tests/ -v
```

You should see output listing each test with **PASSED** next to it. If all tests pass, your installation is working correctly.

**Option B — VS Code Testing panel:**

1. Click the **beaker/flask icon** on the left sidebar (Testing panel)
2. If prompted, click **"Configure Python Tests"** → select **"pytest"** → select the `tests` folder
3. Click the green **▶ Run All Tests** button
4. **Green checkmarks** = passing tests (everything works correctly)
5. **Red X marks** = failing tests (see Troubleshooting in Section 5)

> **If you see `No module named 'pytest'`:** Your virtual environment may not be active or pytest was not installed. Run:
> ```
> venv\Scripts\activate
> python -m pip install -r requirements.txt
> ```

### Step 9: Configure API Keys (Optional)

The tool works out-of-the-box with **mock data** for testing. To use real market data:

1. In the VS Code left sidebar (Explorer), click the **`config`** folder, then click **`settings.yaml`**
2. The file opens in the editor — replace the empty quotes `""` next to each `api_key` with your actual credentials:

```yaml
pjm:
  api_key: "YOUR_PJM_DATAMINER_API_KEY"  # From https://api.pjm.com

gas:
  api_key: "YOUR_MORNINGSTAR_API_KEY"  # Morningstar Commodities credentials
```

3. Press `Ctrl+S` to save

> **Note:** The tool runs fully with mock data when no API keys are set. Mock data is useful for testing and training but should not be used for actual trading decisions.

---

## Section 3b: Friday 3-Day Forecasting

### When to Use Friday Mode

Every Friday morning, PJM's Day-Ahead market clears prices for **Saturday, Sunday, AND Monday all at once**. Your broker or trading desk will quote:

- A **Weekend** Western Hub price for Saturday and Sunday (shared on-peak and off-peak)
- A **Monday** Western Hub price for Monday (separate on-peak and off-peak)
- A single **gas price** that covers the entire Saturday/Sunday/Monday delivery window

### How to Run Friday Mode

Open your VS Code terminal with `(venv)` active and run:

```
python cli.py forecast ^
  --friday-mode ^
  --whub-onpeak-weekend 32.00 ^
  --whub-offpeak-weekend 22.50 ^
  --whub-onpeak-monday 46.00 ^
  --whub-offpeak-monday 28.00 ^
  --gas-price 3.25 ^
  --date 2026-04-11
```

**Flag explanations:**

| Flag | Meaning |
|------|---------|
| `--friday-mode` | Tells the tool to forecast all three days at once |
| `--whub-onpeak-weekend` | The Weekend on-peak power price (HE08–HE23 Sat & Sun) |
| `--whub-offpeak-weekend` | The Weekend off-peak power price (overnight & early morning Sat & Sun) |
| `--whub-onpeak-monday` | Monday's on-peak power price (HE08–HE23 Monday) |
| `--whub-offpeak-monday` | Monday's off-peak power price (HE01–HE07/HE24 Monday) |
| `--gas-price` | One gas price covering all three days |
| `--date` | The Saturday date (first delivery day of the weekend package) |

> **Note for Windows users:** Use the `^` character (caret) to continue a command across multiple lines in Command Prompt, or type the whole command on a single line.

### Refreshing Forecasts on Saturday and Sunday

On Saturday morning, you can **re-run the forecast for Sunday** using the normal (non-Friday) command. This gives a better Sunday forecast because it now uses **real Saturday prices** as D-1 data instead of estimates:

```
python cli.py forecast --whub-onpeak 32.00 --whub-offpeak 22.50 --gas 3.25 --date 2026-04-12
```

On Sunday morning, do the same to refresh the Monday forecast using real Sunday actuals:

```
python cli.py forecast --whub-onpeak 46.00 --whub-offpeak 28.00 --gas 3.25 --date 2026-04-13
```

### Why Sunday and Monday Forecasts Have Wider Confidence Intervals

On Friday, we don't yet know Saturday's actual prices. Sunday's forecast uses Saturday's **predicted** prices as a stand-in for the real D-1 data. This adds uncertainty, which is why the confidence bands are automatically widened:

| Forecast Day | CI Widening | Reason |
|--------------|-------------|--------|
| Saturday | None (standard) | Uses real Friday actuals as D-1 |
| Sunday | +15% wider | Uses Saturday forecast (synthetic) as D-1 |
| Monday | +25% wider | Uses Sunday forecast (double-synthetic) as D-1 |

After running the Saturday refresh on Saturday morning, the Sunday forecast will have standard (non-widened) confidence intervals because it now uses real Saturday actuals.

---

## Section 3: Daily Usage (09:00 EPT)

### Getting Your Input Values

**Western Hub On-Peak & Off-Peak Prices:**
- Check your ICE screen or broker sheet for "PJM Western Hub" Day-Ahead
- Look for "On-Peak" (also written as "5×16" or "peak") and "Off-Peak" (or "7×8" and weekends) prices
- Typical On-Peak range: $30–$80/MWh; Off-Peak: $20–$60/MWh

**Transco Zone 5 Gas Price:**
- Check ICE Natural Gas screen for "Transco Zone 5" next-day gas
- Or call your gas broker for the daily index price
- Typical range: $2.50–$7.00/MMBtu (higher in winter)

### Running the Forecast

Open the VS Code terminal (`` Ctrl+` ``), make sure `(venv)` is showing, and run:

```
python cli.py forecast --whub-onpeak 45.50 --whub-offpeak 28.75 --gas 3.42
```

**Breaking down the command:**
- `python cli.py` — runs the program
- `forecast` — tells it to generate a forecast (not train or evaluate)
- `--whub-onpeak 45.50` — your WHub on-peak price input ($45.50/MWh)
- `--whub-offpeak 28.75` — your WHub off-peak price input ($28.75/MWh)
- `--gas 3.42` — your gas price input ($3.42/MMBtu)

**Optional flags:**
```
# Forecast for a specific date (instead of tomorrow)
python cli.py forecast --whub-onpeak 45.50 --whub-offpeak 28.75 --gas 3.42 --date 2026-01-15

# Save output as CSV
python cli.py forecast --whub-onpeak 45.50 --whub-offpeak 28.75 --gas 3.42 --output csv --output-file forecast.csv

# Save as JSON
python cli.py forecast --whub-onpeak 45.50 --whub-offpeak 28.75 --gas 3.42 --output json
```

### Training the Model (First Time)

Before the model can produce ML-based forecasts, you need to train it once:

```
python cli.py train --start-date 2023-01-01 --end-date 2025-12-31
```

This takes approximately 5–15 minutes and saves the model to `models/ensemble.pkl`. After training, forecasts use the full ML ensemble.

### Scheduling (Optional — Windows Task Scheduler)

1. Open **Task Scheduler** (search in Start menu)
2. Click **Create Basic Task** → name it "PJM SOUTH Forecast"
3. Set trigger: **Daily**, at **9:00 AM**
4. Action: **Start a program**
   - Program: `C:\Users\YourName\pjm-south-lmp-forecast\venv\Scripts\python.exe`
   - Arguments: `cli.py forecast --whub-onpeak 45.50 --whub-offpeak 28.75 --gas 3.42 --output csv --output-file C:\forecasts\today.csv`
   - Start in: `C:\Users\YourName\pjm-south-lmp-forecast`

> **Note:** You'll need to update the price inputs each morning before the scheduled run, or integrate with an automated price data feed.

### Viewing Output Files and Charts

After running the `evaluate` command:
1. Check the **`output/`** folder in the VS Code left sidebar
2. Click on `.png` chart files to view them directly inside VS Code (no external viewer needed)
3. Click on `.csv` files to see tabular data with colored formatting

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
    Gas (Transco Z5): $3.420/MMBtu
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

Python wasn't added to your PATH during installation.

**Fix:** Re-run the Python installer and check **"Add Python to PATH"**, or manually add it:
1. Search **"Environment Variables"** in Start menu
2. Click **"Edit the system environment variables"** → **Environment Variables**
3. Under **User variables**, select `Path` → **Edit** → **New**
4. Add `C:\Users\YourName\AppData\Local\Programs\Python\Python313\` and `C:\Users\YourName\AppData\Local\Programs\Python\Python313\Scripts\`
5. Click **OK** on all dialogs, then **restart** VS Code

### "pip is not recognized" or "No module named pip"

Bare `pip` is not on PATH, or pip is not installed in your Python.

**Fix — always use `python -m pip` instead of bare `pip`:**
```
python -m pip install -r requirements.txt
```

**If that also fails**, bootstrap pip first:
```
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
```

### "ModuleNotFoundError: No module named 'lightgbm'" (or any other module)

The dependencies are not installed, or the virtual environment is not active.

**Fix:**
1. Make sure `(venv)` appears at the start of your terminal prompt. If not:
   ```
   venv\Scripts\activate
   ```
2. Then install dependencies:
   ```
   python -m pip install -r requirements.txt
   ```

### "No module named 'pytest'" when running tests

pytest is listed in `pyproject.toml` under optional dev dependencies and may not have installed.

**Fix:**
```
venv\Scripts\activate
python -m pip install pytest
```
Then run tests with:
```
python -m pytest tests/ -v
```

### "No trained model found" warning

The model hasn't been trained yet. The tool will use a simpler rule-based forecast instead.

**Fix:** Train the model once:
```
python cli.py train --start-date 2023-01-01 --end-date 2025-12-31
```

### API key errors

If you see `NotImplementedError: Set pjm.api_key in config/settings.yaml`, your API key is not configured.

**Fix options:**
1. Add your API key to `config/settings.yaml` (see Section 2, Step 9)
2. Leave the key blank — the tool will automatically use mock data (appropriate for testing only)

### Very slow first run

The first forecast may take 30–60 seconds as Python loads all libraries. Subsequent runs are faster.

### "PermissionError" when saving files

The tool doesn't have write permission to the output directory.

**Fix:** Run from the project directory, or specify a different output path:
```
python cli.py forecast --whub-onpeak 45.0 --whub-offpeak 30.0 --gas 3.5 --output-file C:\Users\YourName\Desktop\forecast.csv
```

---

### VS Code Troubleshooting

| Problem | Solution |
|---------|----------|
| **"Python not found"** in VS Code terminal | Click the Python version in the bottom-left status bar → **Select Interpreter** → choose the `venv` option |
| **Terminal shows `PS C:\...>` and venv activation fails** | PowerShell execution policy issue. Either run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` and retry, **or** switch to Command Prompt: click the **˅** dropdown next to the **+** in the terminal panel → select **Command Prompt** |
| **`"No module named 'click'"` or similar import error** | Make sure `(venv)` appears at the start of your terminal prompt. If not, run `venv\Scripts\activate` then `python -m pip install -r requirements.txt` |
| **`"No module named 'pytest'"` when running tests** | Run `python -m pip install pytest` in the activated venv, or run tests via `python -m pytest tests/ -v` |
| **VS Code shows squiggly red/yellow lines under imports** | Select the correct Python interpreter (Section 2, Step 7) — choose the `venv` one |
| **Charts don't display when clicking `.png` files** | Install the **"Image Preview"** extension: click Extensions icon → search "Image Preview" → Install |
| **Terminal output is too small to read** | Drag the **top edge** of the terminal panel upward to make it taller |
| **VS Code asks "Do you trust the authors of this folder?"** | Click **"Yes, I trust the authors"** — this is your own project folder |
| **`(venv)` disappears after reopening VS Code** | Re-activate with `venv\Scripts\activate` each session, or add `"python.terminal.activateEnvironment": true` in VS Code settings (`Ctrl+,` → search "activate") |

---

## Section 6: Quick Reference Card

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PJM SOUTH DA LMP FORECAST — QUICK REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SETUP (one time, in VS Code terminal):
  python -m venv venv
  venv\Scripts\activate
  python -m pip install -r requirements.txt

VERIFY INSTALLATION:
  python -m pytest tests/ -v

DAILY COMMAND (09:00 EPT):
  python cli.py forecast ^
    --whub-onpeak  [PRICE]  ^   ← WHub On-Peak $/MWh
    --whub-offpeak [PRICE]  ^   ← WHub Off-Peak $/MWh
    --gas          [PRICE]      ← Transco Z5 $/MMBtu

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
  HE08–HE23, Monday–Friday, excluding NERC holidays

NERC HOLIDAYS (2026):
  Jan 1, May 25, Jul 4, Sep 7, Nov 26, Dec 25

SPIKE RISK THRESHOLDS:
  Low       < 10%  │  Moderate  10–35%
  High    35–65%   │  Very High  > 65%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

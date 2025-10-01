# How to Use the Combined Odds Scraper in Your Notebook

## Step-by-Step Instructions:

### 1. Open Your Notebook
- Open `notebooks/01_Fight_Predictor_Pipeline.ipynb`

### 2. Find the "## Odds Scrapper" Section
- Look for the markdown cell that says "## Odds Scrapper"
- You should see an empty code cell right after it

### 3. Copy the Code
- Open the file `notebooks/odds_scraper_notebook_code.py`
- Copy ALL the code from that file

### 4. Paste into the Empty Cell
- Click on the empty code cell after "## Odds Scrapper"
- Paste the copied code
- The cell should now contain the combined odds scraper code

### 5. Run the Cell
- Press `Shift + Enter` to run the cell
- You should see output showing:
  - Data processing progress
  - Odds clamping results
  - ROI analysis
  - Monthly breakdown

### 6. Use the Processed Data
- After running, you'll have a `df_with_odds` variable available
- This contains all your fight data with cleaned odds
- You can use this DataFrame in subsequent cells

## What the Code Does:

✅ **Fixes Data Quality Issues:**
- Clamps extreme odds (102 → 100, -105 → -100)
- Handles missing odds intelligently
- Filters unrealistic odds ranges

✅ **Provides Analysis:**
- Shows realistic ROI calculations
- Monthly performance breakdown
- Win rates and profit/loss

✅ **Returns Clean Data:**
- `df_with_odds` DataFrame ready for model training
- All odds data properly processed
- No more 428% ROI anomalies!

## Expected Output:
```
=== COMBINED ODDS SCRAPER ===
This processor combines:
• Odds clamping (fixes extreme values like 102 → 100)
• Improved filtering (handles missing odds intelligently)
• ROI analysis with realistic calculations

=== SIMPLE ODDS PROCESSOR ===
Loading data from: ../src/final_with_odds.csv
Processing 1482 fights

=== APPLYING DATA QUALITY FIXES ===
=== APPLYING ODDS CLAMPING ===
  draftkings_odds: Clamped 84 values
  fanduel_odds: Clamped 80 values
  betmgm_odds: Clamped 90 values
  bovada_odds: Clamped 59 values

=== RESULTS ===
✅ Processed 1482 fights
✅ ROI: -16.36%
✅ Win rate: 29.97%
✅ Total profit: $-9,880.63
```

## Troubleshooting:

**If you get import errors:**
- Make sure the `src/simple_odds_processor.py` file exists
- Check that your notebook is in the `notebooks/` folder

**If you get file not found errors:**
- Make sure `../src/final_with_odds.csv` exists
- Check your file paths are correct

**If you want to use different data:**
- Change the `input_csv` parameter in the `process_odds_data()` call
- Make sure the CSV has the required columns: DATE, EVENT, BOUT, FIGHTER, win, and odds columns

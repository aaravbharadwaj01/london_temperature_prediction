# London Temperature Prediction Dashboard

A dark dashboard for project ML_73, available in both Streamlit and React. Both versions read the notebook's supplied prediction and evaluation outputs and do not retrain models or manufacture scores.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder and run:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   streamlit run app.py
   ```

The browser opens the dashboard. Keep `data/` beside `app.py`; the four CSVs there are the original uploaded outputs copied without editing.

## Run the React version

The React app is in `react-dashboard/` and uses the same uploaded CSV outputs. Install Node.js 18 or newer, then run:

```bash
cd react-dashboard
npm install
npm run dev
```

Open the local URL printed by Vite. To create a production build, run `npm run build`; the generated site is in `react-dashboard/dist/`.

## What it shows

- Notebook monthly averages and saved Random Forest feature importances.
- Daily actual-versus-predicted test-set views and supplied model metrics.
- Five-fold walk-forward results and fold-level scores.
- A filterable and downloadable prediction-output table.
- A separate responsive React interface with the same project sections and results.

## Data and prediction limits

The supplied prediction CSV contains historical test-set rows, from 2023 onward. The app visualizes those stored predictions; it does not make live or future forecasts. The notebook reports a full raw dataset from 2010-01-01 to 2025-12-31 (5,844 daily records), but that raw daily file was not among the provided uploads. Full-history facts and monthly averages displayed here come from notebook outputs; the explorer exposes the supplied test-set CSV.

The model comparison CSV includes Linear Regression, Random Forest, and Decision Tree. The prediction and walk-forward files also contain a Seasonal Naive baseline. The dashboard preserves this difference between the supplied outputs.

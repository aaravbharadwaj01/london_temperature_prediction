from pathlib import Path
from datetime import timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).parent
DATA = ROOT / "data" if (ROOT / "data" / "ML_73_temperature_predictions.csv").exists() else ROOT
RAW_DATA_NAMES = ("london_temperature_2010_2025.csv",)
COLORS = {"Linear Regression": "#60a5fa", "Random Forest": "#a78bfa", "Decision Tree": "#fbbf24", "Seasonal Naive": "#94a3b8"}
PREDICTION_COLUMNS = {
    "Linear Regression": "linear_regression",
    "Random Forest": "random_forest",
    "Decision Tree": "decision_tree",
    "Seasonal Naive": "seasonal_naive",
}
FEATURE_IMPORTANCE = pd.DataFrame({
    "Feature": ["lag_1", "year_cos", "rolling_mean_7", "rolling_mean_30", "lag_2", "lag_365", "lag_7", "lag_30", "lag_14", "lag_3", "day", "day_of_year", "year_sin", "year", "day_of_week", "month"],
    "Importance": [0.907385, 0.009127, 0.008420, 0.008158, 0.008018, 0.007971, 0.007637, 0.007183, 0.006809, 0.006600, 0.005385, 0.005302, 0.004085, 0.003974, 0.003609, 0.000338],
})

FEATURE_COLS = [
    "year", "month", "day", "day_of_week", "day_of_year", "year_sin", "year_cos",
    "lag_1", "lag_2", "lag_3", "lag_7", "lag_14", "lag_30", "lag_365",
    "rolling_mean_7", "rolling_mean_30",
]


def make_features(history):
    """Recreate the notebook's calendar, lag, and rolling-average features."""
    frame = history.copy().sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month
    frame["day"] = frame["date"].dt.day
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["day_of_year"] = frame["date"].dt.dayofyear
    frame["year_sin"] = np.sin(2 * np.pi * frame["day_of_year"] / 365.25)
    frame["year_cos"] = np.cos(2 * np.pi * frame["day_of_year"] / 365.25)
    for lag in [1, 2, 3, 7, 14, 30, 365]:
        frame[f"lag_{lag}"] = frame["mean_temp"].shift(lag)
    frame["rolling_mean_7"] = frame["mean_temp"].shift(1).rolling(7).mean()
    frame["rolling_mean_30"] = frame["mean_temp"].shift(1).rolling(30).mean()
    return frame


@st.cache_resource(show_spinner="Training models on the daily temperature history…")
def train_forecast_models(history):
    featured = make_features(history).dropna(subset=FEATURE_COLS + ["mean_temp"])
    x, y = featured[FEATURE_COLS], featured["mean_temp"]
    models_for_forecast = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_leaf=3, random_state=42, n_jobs=-1),
    }
    for fitted_model in models_for_forecast.values():
        fitted_model.fit(x, y)
    return models_for_forecast


def forecast_to_date(history, target_date, model_name, fitted_models):
    """Recursively predict one day at a time using prior observed/predicted values."""
    series = history.set_index("date")["mean_temp"].copy().sort_index()
    last_observed = series.index.max()
    target_date = pd.Timestamp(target_date)
    for forecast_date in pd.date_range(last_observed + pd.Timedelta(days=1), target_date, freq="D"):
        prior = series.loc[series.index < forecast_date]
        if len(prior) < 365:
            raise ValueError("At least 365 consecutive daily temperature values are needed to forecast.")
        day_of_year = forecast_date.dayofyear
        row = {
            "year": forecast_date.year, "month": forecast_date.month, "day": forecast_date.day,
            "day_of_week": forecast_date.dayofweek, "day_of_year": day_of_year,
            "year_sin": np.sin(2 * np.pi * day_of_year / 365.25),
            "year_cos": np.cos(2 * np.pi * day_of_year / 365.25),
            "lag_1": prior.iloc[-1], "lag_2": prior.iloc[-2], "lag_3": prior.iloc[-3],
            "lag_7": prior.iloc[-7], "lag_14": prior.iloc[-14], "lag_30": prior.iloc[-30],
            "lag_365": prior.iloc[-365], "rolling_mean_7": prior.iloc[-7:].mean(),
            "rolling_mean_30": prior.iloc[-30:].mean(),
        }
        if model_name == "Seasonal Naive":
            prediction = float(row["lag_365"])
        else:
            prediction = float(fitted_models[model_name].predict(pd.DataFrame([row], columns=FEATURE_COLS))[0])
        series.loc[forecast_date] = prediction
    forecast_path = series.loc[series.index > last_observed].rename("Predicted temperature").reset_index()
    return float(series.loc[target_date]), last_observed.date(), forecast_path


def find_default_history():
    candidates = [ROOT / "data", ROOT]
    for filename in RAW_DATA_NAMES:
        for folder in candidates:
            candidate = folder / filename
            if candidate.exists():
                return candidate
    return None


@st.cache_data
def load_default_history():
    path = find_default_history()
    if path is None:
        return pd.DataFrame(columns=["date", "mean_temp", "min_temp", "max_temp"])
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame.sort_values("date").reset_index(drop=True)

st.set_page_config(page_title="London Temperature Lab", page_icon="🌡️", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --ink:#f4f6fb; --muted:#9aa7bc; --panel:#111b2b; --line:#223149; --accent:#7dd3fc; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: radial-gradient(ellipse at 85% 0%, #142c49 0%, #0a1220 42%, #080e18 100%); color:var(--ink); }
[data-testid="stSidebar"] { background:#0c1523; border-right:1px solid #1c2a3d; }
[data-testid="stMetric"] { background:linear-gradient(145deg,#152338,#101a29); border:1px solid #263852; padding:18px 20px; border-radius:16px; }
[data-testid="stMetricLabel"] { color:#aab7ca; }
[data-testid="stMetricValue"] { color:#f4f6fb; font-family:'Space Grotesk',sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.03em; }
.eyebrow { color:#7dd3fc; text-transform:uppercase; letter-spacing:.15em; font-weight:700; font-size:.74rem; }
.hero { padding:16px 0 8px 0; }
.note { background:#142338; border:1px solid #29425f; border-left:4px solid #7dd3fc; padding:14px 18px; border-radius:10px; color:#ced9e8; }
.small-muted { color:#9aa7bc; font-size:.9rem; }
div[data-testid="stPlotlyChart"] { border:1px solid #1e2c40; border-radius:15px; background:rgba(15,24,38,.65); padding:8px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    predictions = pd.read_csv(DATA / "ML_73_temperature_predictions.csv", parse_dates=["date"])
    comparison = pd.read_csv(DATA / "ML_73_model_comparison.csv")
    # The uploaded export uses a replacement glyph in its temperature-unit headers.
    comparison = comparison.rename(columns={c: "MAE (°C)" if c.lower().startswith("mae") else "RMSE (°C)" if c.lower().startswith("rmse") else c for c in comparison.columns})
    folds = pd.read_csv(DATA / "ML_73_walk_forward_fold_results.csv")
    summary = pd.read_csv(DATA / "ML_73_walk_forward_summary.csv")
    return predictions, comparison, folds, summary


pred, comparison, folds, summary = load_data()
history_data = load_default_history()
models = [name for name, col in PREDICTION_COLUMNS.items() if col in pred.columns]
best = comparison.sort_values("MAE (°C)").iloc[0]
date_min, date_max = pred.date.min().date(), pred.date.max().date()

with st.sidebar:
    st.markdown("## 🌡️ London Temperature Lab")
    st.caption("ML project · ML_73 · 5CIUP521")
    section = st.radio("Explore", ["Overview", "Prediction lookup", "Future estimate", "Temperature trends", "Model performance", "Walk-forward validation", "Dataset explorer"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Prediction period**")
    st.caption(f"{date_min:%d %b %Y} — {date_max:%d %b %Y}")
    st.caption("Saved results + future estimates")

st.markdown('<div class="hero"><div class="eyebrow">London · 51.5072°N, 0.1276°W</div><h1>Temperature prediction dashboard</h1><p class="small-muted">A closer look at historical weather patterns and time-series model performance.</p></div>', unsafe_allow_html=True)

if section == "Overview":
    st.markdown('<div class="note">The overview now uses the complete 2010–2025 daily history. Model scores and actual-versus-predicted results below remain the notebook’s original held-out test-set outputs.</div>', unsafe_allow_html=True)
    st.write("")
    c1, c2, c3, c4, c5 = st.columns(5)
    if not history_data.empty:
        c1.metric("Historical daily records", f"{len(history_data):,}")
        mean_history_temp = float(history_data["mean_temp"].mean())
        c2.metric("Full-history mean", f"{mean_history_temp:.2f} °C")
        c5.metric("Historical period", f"{history_data.date.min():%Y}–{history_data.date.max():%Y}")
    else:
        c1.metric("Historical daily records", "Unavailable")
        c2.metric("Full-history mean", "Unavailable")
        c5.metric("Historical period", "Unavailable")
        mean_history_temp = None
    c3.metric("Best test MAE", f"{best['MAE (°C)']:.3f} °C", str(best["Model"]))
    c4.metric("Best model R²", f"{best['R2 Score']:.3f}")
    if mean_history_temp is not None:
        st.subheader("Daily mean temperature · full dataset")
        history_fig = px.line(history_data, x="date", y="mean_temp", labels={"date": "", "mean_temp": "Mean temperature (°C)"})
        history_fig.update_traces(line_color="#7dd3fc", line_width=1)
        history_fig.update_layout(height=390, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(history_fig, use_container_width=True)
        st.caption(f"All {len(history_data):,} daily observations from {history_data.date.min():%d %b %Y} through {history_data.date.max():%d %b %Y}.")
    st.subheader("Actual and predicted temperatures · test set")
    selected = st.selectbox("Prediction model", models, index=models.index("Linear Regression") if "Linear Regression" in models else 0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred.date, y=pred.actual_temperature, name="Actual", line=dict(color="#e2e8f0", width=1.5)))
    fig.add_trace(go.Scatter(x=pred.date, y=pred[PREDICTION_COLUMNS[selected]], name=selected, line=dict(color=COLORS[selected], width=1.5)))
    fig.update_layout(height=400, margin=dict(l=20,r=20,t=20,b=20), template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Mean temperature (°C)", xaxis_title="", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("The prediction CSV contains the chronological test-set results, beginning in 2023. It is not the full raw 2010–2025 weather series.")
    left, right = st.columns(2)
    with left:
        st.subheader("Test-set model comparison")
        chart = px.bar(comparison.sort_values("MAE (°C)"), x="Model", y="MAE (°C)", color="Model", color_discrete_map=COLORS, text_auto=".3f")
        chart.update_layout(template="plotly_dark", height=330, showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=20,b=10), yaxis_title="MAE (°C)")
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.subheader("Notebook monthly averages")
        monthly = pd.DataFrame({"Month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], "Mean °C": [4.53,5.46,7.04,9.54,12.86,16.22,18.30,17.69,15.24,12.16,8.10,6.14]})
        chart = px.line(monthly, x="Month", y="Mean °C", markers=True)
        chart.update_traces(line_color="#7dd3fc", marker_color="#7dd3fc")
        chart.update_layout(template="plotly_dark", height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(chart, use_container_width=True)
        st.caption("Computed from the notebook’s full 2010–2025 dataset.")

elif section == "Prediction lookup":
    st.subheader("Look up a saved prediction")
    st.markdown('<div class="note">Choose a date from the historical test set and a model to see the prediction saved by your notebook. This lookup does not generate a new or future forecast.</div>', unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    with a:
        chosen_date = st.date_input("Select a test-set date", value=date_min, min_value=date_min, max_value=date_max)
    with b:
        chosen_model = st.selectbox("Choose a model", models, index=models.index("Linear Regression") if "Linear Regression" in models else 0, key="lookup_model")
    match = pred.loc[pred.date.dt.date == chosen_date]
    if match.empty:
        st.warning("No saved prediction exists for that date.")
    else:
        row = match.iloc[0]
        prediction = float(row[PREDICTION_COLUMNS[chosen_model]])
        actual = float(row["actual_temperature"])
        error = prediction - actual
        st.markdown(f"### {chosen_date:%A, %d %B %Y}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Model prediction", f"{prediction:.2f} °C", chosen_model)
        c2.metric("Recorded temperature", f"{actual:.2f} °C")
        c3.metric("Absolute error", f"{abs(error):.2f} °C", f"{'+' if error >= 0 else '−'}{error:.2f} °C prediction error")
        st.subheader("All saved model outputs for this date")
        view = pd.DataFrame({"Model": models, "Prediction (°C)": [float(row[PREDICTION_COLUMNS[m]]) for m in models]})
        st.dataframe(view.style.format({"Prediction (°C)": "{:.3f}"}), use_container_width=True, hide_index=True)
        st.caption("The actual and predicted values are read from the supplied historical test-set CSV. Available dates: 1 Jan 2023 to 31 Dec 2025.")

elif section == "Future estimate":
    st.subheader("Estimate a future temperature")
    st.markdown('<div class="note">Choose a past or future date and a model. Past dates use a one-day backtest trained on data available before that date. Future dates are predicted one day at a time using earlier estimates as lag features.</div>', unsafe_allow_html=True)
    default_history_path = find_default_history()
    uploaded_history = st.file_uploader("Optional: upload an updated daily history CSV", type=["csv"], help="Required columns: date and mean_temp. Include at least 365 consecutive daily observations; min_temp and max_temp are optional.")
    history = None
    try:
        if uploaded_history is not None:
            history = pd.read_csv(BytesIO(uploaded_history.getvalue()))
            history_source = f"Uploaded file: {uploaded_history.name}"
        elif default_history_path is not None:
            history = pd.read_csv(default_history_path)
            history_source = default_history_path.name
        if history is None:
            st.warning(f"No bundled history file was found. Upload a daily temperature CSV with date and mean_temp columns. Expected default: {RAW_DATA_NAMES[0]}.")
        else:
            if "date" not in history.columns and "time" in history.columns:
                history = history.rename(columns={"time": "date"})
            if "mean_temp" not in history.columns and "temperature_2m_mean" in history.columns:
                history = history.rename(columns={"temperature_2m_mean": "mean_temp"})
            if not {"date", "mean_temp"}.issubset(history.columns):
                st.error("The CSV must contain columns named date and mean_temp (or time and temperature_2m_mean).")
            else:
                history["date"] = pd.to_datetime(history["date"], errors="coerce")
                history["mean_temp"] = pd.to_numeric(history["mean_temp"], errors="coerce")
                history = history.dropna(subset=["date", "mean_temp"])[["date", "mean_temp"]].drop_duplicates("date").sort_values("date").reset_index(drop=True)
                expected = pd.date_range(history.date.min(), history.date.max(), freq="D")
                if len(history) < 400:
                    st.error("Please provide at least 400 daily records; the model needs one year of history to create its lag features.")
                elif not history.date.equals(pd.Series(expected, name="date")):
                    st.error("The temperature history must have one continuous row per day. Fill missing dates before forecasting.")
                else:
                    latest = history.date.max().date()
                    age_days = (pd.Timestamp.today().date() - latest).days
                    if age_days > 30:
                        st.warning(f"This history ends on {latest:%d %b %Y}, which is {age_days} days ago. Future estimates start from that last observation; historical backtests use only earlier rows.")
                    st.caption(f"Using {len(history):,} daily records · {history_source} · last observation {latest:%d %b %Y}")
                    target_date = st.date_input("Target date (past or future)", value=pd.Timestamp.today().date() + timedelta(days=1))
                    model_name = st.selectbox("Forecast model", ["Linear Regression", "Random Forest", "Decision Tree", "Seasonal Naive"])
                    target_timestamp = pd.Timestamp(target_date)
                    is_historical = target_timestamp <= history.date.max()
                    if not is_historical:
                        horizon = (target_date - latest).days
                        if horizon > 30:
                            st.warning(f"This estimate is {horizon} days beyond the last observation. Recursive errors can compound as the horizon grows.")
                        if horizon > 3650:
                            st.error("For app responsiveness, future estimates are limited to 10 years beyond the last observation.")
                    if st.button("Estimate temperature", type="primary"):
                        try:
                            if is_historical:
                                training_history = history.loc[history.date < target_timestamp].copy()
                                if len(training_history) < 730:
                                    earliest_supported = (history.date.min() + pd.Timedelta(days=730)).date()
                                    st.error(f"This model needs at least two years of earlier daily data. The earliest supported date for this file is {earliest_supported:%d %b %Y}.")
                                    st.stop()
                                mode_label = "Historical one-day backtest"
                            else:
                                training_history = history
                                mode_label = "Future estimate"
                                if horizon > 3650:
                                    st.stop()
                            fitted = train_forecast_models(training_history) if model_name != "Seasonal Naive" else {}
                            estimate, last_observed, path = forecast_to_date(training_history, target_date, model_name, fitted)
                            st.markdown(f"### Estimate for {target_date:%A, %d %B %Y}")
                            st.caption(f"{mode_label} · {model_name}")
                            if is_historical:
                                actual_rows = history.loc[history.date == target_timestamp, "mean_temp"]
                                actual = float(actual_rows.iloc[0]) if not actual_rows.empty else None
                                x1, x2, x3 = st.columns(3)
                                x1.metric("Model estimate", f"{estimate:.2f} °C")
                                if actual is not None:
                                    x2.metric("Recorded temperature", f"{actual:.2f} °C")
                                    x3.metric("Absolute error", f"{abs(estimate - actual):.2f} °C")
                                else:
                                    x2.metric("Training data through", f"{last_observed:%d %b %Y}")
                            else:
                                x1, x2, x3 = st.columns(3)
                                x1.metric("Estimated mean temperature", f"{estimate:.2f} °C", model_name)
                                x2.metric("Forecast horizon", f"{horizon} day{'s' if horizon != 1 else ''}")
                                x3.metric("Last observed date", f"{last_observed:%d %b %Y}")
                            fig = go.Figure()
                            observed_tail = training_history.tail(90)
                            fig.add_trace(go.Scatter(x=observed_tail.date, y=observed_tail.mean_temp, name="Observed history", line=dict(color="#e2e8f0", width=1.5)))
                            fig.add_trace(go.Scatter(x=path["date"], y=path["Predicted temperature"], name="Estimated temperature", line=dict(color=COLORS.get(model_name, "#94a3b8"), width=1.5, dash="dash")))
                            if is_historical and actual is not None:
                                fig.add_trace(go.Scatter(x=[target_timestamp], y=[actual], name="Recorded temperature", mode="markers", marker=dict(color="#50d5a2", size=9)))
                            fig.update_layout(height=400, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis_title="Mean temperature (°C)", xaxis_title="", margin=dict(l=20,r=20,t=20,b=20))
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption("Historical dates use a one-day backtest trained only on earlier observations. Future dates are predicted recursively from the last observation. Neither mode uses atmospheric conditions, so results are estimates rather than weather forecasts.")
                        except Exception as exc:
                            st.error(f"Could not create the estimate: {exc}")
    except Exception as exc:
        st.error(f"Could not read the temperature history: {exc}")

elif section == "Temperature trends":
    st.subheader("Historical patterns")
    st.markdown("Monthly climatology below comes from the notebook’s complete dataset. The daily chart uses the supplied prediction CSV, which covers only its held-out test period.")
    monthly = pd.DataFrame({"Month": ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], "Mean °C": [4.53,5.46,7.04,9.54,12.86,16.22,18.30,17.69,15.24,12.16,8.10,6.14]})
    fig=px.line(monthly,x="Month",y="Mean °C",markers=True); fig.update_traces(line_color="#7dd3fc",marker_color="#7dd3fc",line_width=3)
    fig.update_layout(template="plotly_dark",height=370,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    y1,y2=st.select_slider("Daily test-set period",options=sorted(pred.date.dt.year.unique()),value=(int(pred.date.dt.year.min()),int(pred.date.dt.year.max())))
    d=pred[pred.date.dt.year.between(y1,y2)]
    fig=px.line(d,x="date",y="actual_temperature",labels={"date":"","actual_temperature":"Actual mean temperature (°C)"})
    fig.update_traces(line_color="#e2e8f0",line_width=1)
    fig.update_layout(template="plotly_dark",height=390,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    m=d.assign(month=d.date.dt.month_name()).groupby("month",sort=False).actual_temperature.agg(["mean","min","max"]).reset_index()
    st.dataframe(m.rename(columns={"month":"Month","mean":"Mean °C","min":"Min °C","max":"Max °C"}).style.format({"Mean °C":"{:.2f}","Min °C":"{:.2f}","Max °C":"{:.2f}"}),use_container_width=True,hide_index=True)

elif section == "Model performance":
    st.subheader("Held-out test-set comparison")
    st.caption("Metrics are reproduced from the notebook output CSV. Lower MAE and RMSE are better; R² closer to 1 indicates a stronger fit.")
    metric=st.selectbox("Compare metric", ["MAE (°C)","RMSE (°C)","R2 Score"])
    fig=px.bar(comparison.sort_values(metric),x="Model",y=metric,color="Model",color_discrete_map=COLORS,text_auto=".4f")
    fig.update_layout(template="plotly_dark",height=390,showlegend=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(comparison.style.format({c:"{:.4f}" for c in comparison.select_dtypes("number").columns}),use_container_width=True,hide_index=True)
    st.subheader("Random Forest feature importance")
    st.caption("Values transcribed from the saved notebook output; importance describes this fitted model and is not a causal effect.")
    imp=FEATURE_IMPORTANCE.sort_values("Importance")
    fig=px.bar(imp,x="Importance",y="Feature",orientation="h",color="Importance",color_continuous_scale=["#315879","#7dd3fc"])
    fig.update_layout(template="plotly_dark",height=480,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",coloraxis_showscale=False)
    st.plotly_chart(fig,use_container_width=True)
    st.subheader("Prediction detail")
    model=st.selectbox("Overlay model",models,key="performance_model")
    fig=go.Figure([go.Scatter(x=pred.date,y=pred.actual_temperature,name="Actual",line=dict(color="#e2e8f0")),go.Scatter(x=pred.date,y=pred[PREDICTION_COLUMNS[model]],name=model,line=dict(color=COLORS[model]))])
    fig.update_layout(template="plotly_dark",height=400,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",yaxis_title="Temperature (°C)")
    st.plotly_chart(fig,use_container_width=True)
    st.info("Linear Regression has the lowest test-set MAE in the supplied model-comparison file. The seasonal-naive baseline is present in the prediction and walk-forward files, but not in the supplied test-set comparison CSV.")

elif section == "Walk-forward validation":
    st.subheader("Five-fold chronological validation")
    st.caption("Each fold trains on earlier dates and evaluates on a later block. Summary values below are the supplied outputs.")
    metric=st.selectbox("Summary metric",["Mean_MAE","Mean_RMSE","Mean_R2"])
    fig=px.bar(summary.sort_values(metric),x="Model",y=metric,color="Model",color_discrete_map=COLORS,text_auto=".4f",error_y="Std_MAE" if metric=="Mean_MAE" else None)
    fig.update_layout(template="plotly_dark",height=380,showlegend=False,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(summary.style.format({c:"{:.4f}" for c in summary.select_dtypes("number").columns}),use_container_width=True,hide_index=True)
    st.subheader("Scores by fold")
    fold_metric=st.selectbox("Fold metric",["MAE","RMSE","R2"])
    fig=px.line(folds,x="Fold",y=fold_metric,color="Model",markers=True,color_discrete_map=COLORS)
    fig.update_layout(template="plotly_dark",height=390,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(folds,use_container_width=True,hide_index=True)
    st.info("Across the five folds, Random Forest has the lowest mean MAE in the supplied summary; the test-set comparison separately ranks Linear Regression first. These are different evaluation procedures and outcomes.")

elif section == "Dataset explorer":
    st.subheader("Prediction output explorer")
    st.markdown("Browse the saved test-set predictions and the full daily temperature history used for the interactive estimate.")
    years=sorted(pred.date.dt.year.unique())
    selected_years=st.multiselect("Filter years",years,default=years)
    view=pred[pred.date.dt.year.isin(selected_years)].copy()
    st.metric("Rows shown",f"{len(view):,}")
    st.dataframe(view,use_container_width=True,hide_index=True)
    st.download_button("Download filtered rows",view.to_csv(index=False).encode("utf-8"),"london_temperature_test_predictions.csv","text/csv")
    raw_path = find_default_history()
    if raw_path is not None:
        st.subheader("Full daily history")
        raw = pd.read_csv(raw_path, parse_dates=["date"])
        raw_years = sorted(raw.date.dt.year.unique())
        raw_selected_years = st.multiselect("Filter history years", raw_years, default=raw_years, key="history_year_filter")
        raw_view = raw[raw.date.dt.year.isin(raw_selected_years)]
        st.caption(f"{len(raw_view):,} daily rows · {raw_path.name}")
        st.dataframe(raw_view, use_container_width=True, hide_index=True)
        st.download_button("Download filtered history", raw_view.to_csv(index=False).encode("utf-8"), "london_temperature_history.csv", "text/csv")
    st.subheader("Notebook dataset facts")
    a,b,c=st.columns(3); a.metric("Raw daily observations","5,844"); b.metric("Notebook date range","2010–2025"); c.metric("Mean temperature range","−6.4 to 29.1 °C")
    st.caption("The full-series facts come from the notebook and the bundled daily history file. Test-set prediction results remain the original supplied outputs.")
    st.subheader("Random Forest feature importance")
    st.caption("Copied from the notebook’s saved output; importance is model-specific and not causal.")
    imp=FEATURE_IMPORTANCE.sort_values("Importance")
    fig=px.bar(imp,x="Importance",y="Feature",orientation="h",color="Importance",color_continuous_scale=["#315879","#7dd3fc"])
    fig.update_layout(template="plotly_dark",height=480,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",coloraxis_showscale=False)
    st.plotly_chart(fig,use_container_width=True)

st.divider()
st.caption("Project ML_73 · Predicting Temperature in London · Open-Meteo Archive historical data · Notebook and supplied CSV outputs")

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).parent
DATA = ROOT / "data"
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
models = [name for name, col in PREDICTION_COLUMNS.items() if col in pred.columns]
best = comparison.sort_values("MAE (°C)").iloc[0]
date_min, date_max = pred.date.min().date(), pred.date.max().date()

with st.sidebar:
    st.markdown("## 🌡️ London Temperature Lab")
    st.caption("ML project · ML_73 · 5CIUP521")
    section = st.radio("Explore", ["Overview", "Temperature trends", "Model performance", "Walk-forward validation", "Dataset explorer"], label_visibility="collapsed")
    st.divider()
    st.markdown("**Prediction period**")
    st.caption(f"{date_min:%d %b %Y} — {date_max:%d %b %Y}")
    st.caption("Historical test-set outputs")

st.markdown('<div class="hero"><div class="eyebrow">London · 51.5072°N, 0.1276°W</div><h1>Temperature prediction dashboard</h1><p class="small-muted">A closer look at historical weather patterns and time-series model performance.</p></div>', unsafe_allow_html=True)

if section == "Overview":
    st.markdown('<div class="note">This dashboard reports the notebook’s saved historical test-set predictions. It does not generate a forecast for future dates. Values and scores below come directly from the supplied CSV outputs.</div>', unsafe_allow_html=True)
    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test-set days", f"{len(pred):,}")
    c2.metric("Best test MAE", f"{best['MAE (°C)']:.3f} °C", str(best["Model"]))
    c3.metric("Best model R²", f"{best['R2 Score']:.3f}")
    c4.metric("Test period", f"{pred.date.dt.year.min()}–{pred.date.dt.year.max()}")
    st.subheader("Actual and predicted temperatures")
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
    st.markdown("The supplied prediction file contains **daily actual and predicted temperatures for the test set**, not the raw daily weather dataset used for training. Notebook-level full-series details are shown separately below.")
    years=sorted(pred.date.dt.year.unique())
    selected_years=st.multiselect("Filter years",years,default=years)
    view=pred[pred.date.dt.year.isin(selected_years)].copy()
    st.metric("Rows shown",f"{len(view):,}")
    st.dataframe(view,use_container_width=True,hide_index=True)
    st.download_button("Download filtered rows",view.to_csv(index=False).encode("utf-8"),"london_temperature_test_predictions.csv","text/csv")
    st.subheader("Notebook dataset facts")
    a,b,c=st.columns(3); a.metric("Raw daily observations","5,844"); b.metric("Notebook date range","2010–2025"); c.metric("Mean temperature range","−6.4 to 29.1 °C")
    st.caption("The raw 5,844-row weather dataset was not among the supplied CSVs. The dates and summary facts above are reported by the notebook, and the prediction CSV is the browsable dataset included here.")
    st.subheader("Random Forest feature importance")
    st.caption("Copied from the notebook’s saved output; the raw fitting dataset is not bundled.")
    imp=FEATURE_IMPORTANCE.sort_values("Importance")
    fig=px.bar(imp,x="Importance",y="Feature",orientation="h",color="Importance",color_continuous_scale=["#315879","#7dd3fc"])
    fig.update_layout(template="plotly_dark",height=480,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",coloraxis_showscale=False)
    st.plotly_chart(fig,use_container_width=True)

st.divider()
st.caption("Project ML_73 · Predicting Temperature in London · Open-Meteo Archive historical data · Notebook and supplied CSV outputs")

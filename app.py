import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from io import BytesIO
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smarty Stocks Pro",
    page_icon="📦",
    layout="wide"
)

# =====================================================
# SESSION DEFAULTS
# =====================================================
DEFAULTS = {
    "page": "Dashboard",
    "trees": 120,
    "test_ratio": 0.20,
    "anomaly": 0.05,
    "uploaded_file_bytes": None,
    "uploaded_file_name": None,
    "selected_stores": None,
    "selected_categories": None,
    "date_range": None,
    "clip_charts": True,
    "clip_percentile": 95,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>
.stApp{
    background:#f6f8fc;
}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0b1736 0%,#12244d 100%);
    border-right:1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div{
    color:#eef2ff !important;
}

section[data-testid="stSidebar"] .stButton > button{
    width:100%;
    border-radius:14px !important;
    font-weight:700 !important;
    padding:0.85rem 0.95rem !important;
    border:1px solid rgba(255,255,255,0.10) !important;
    background:rgba(255,255,255,0.08) !important;
    color:#eef2ff !important;
}

section[data-testid="stSidebar"] .stButton > button:hover{
    background:rgba(255,255,255,0.16) !important;
}

.main-title{
    font-size:2.2rem;
    font-weight:800;
    color:#172033;
    margin-bottom:0.1rem;
}

.sub-title{
    color:#6b7280;
    font-size:0.96rem;
    margin-bottom:1rem;
}

.kpi-card{
    background:#ffffff;
    border-radius:18px;
    padding:1rem;
    border:1px solid #e7ecf5;
    box-shadow:0 8px 24px rgba(16,24,40,0.05);
    text-align:center;
    min-height:110px;
}

.kpi-label{
    color:#7b8798;
    font-size:0.9rem;
    margin-bottom:0.45rem;
    font-weight:600;
}

.kpi-value{
    font-size:1.75rem;
    font-weight:800;
    color:#162033;
}

.panel{
    background:#ffffff;
    border-radius:18px;
    padding:1rem;
    border:1px solid #e7ecf5;
    box-shadow:0 8px 24px rgba(16,24,40,0.05);
}

.info-box{
    background:#f8fbff;
    border-left:4px solid #3b82f6;
    padding:1rem 1rem;
    border-radius:12px;
    color:#334155;
    line-height:1.6;
    border-top:1px solid #e7ecf5;
    border-right:1px solid #e7ecf5;
    border-bottom:1px solid #e7ecf5;
}

.summary-box{
    background:#ffffff;
    padding:0.95rem 1rem;
    border-radius:16px;
    border:1px solid #e7ecf5;
    box-shadow:0 8px 24px rgba(16,24,40,0.05);
    text-align:right;
    color:#1f2a44;
    font-size:0.94rem;
}

.section-chip{
    display:inline-block;
    background:#2563eb;
    color:white;
    padding:0.45rem 0.95rem;
    border-radius:999px;
    font-weight:700;
    font-size:0.84rem;
    margin-bottom:0.75rem;
}

.stDownloadButton > button{
    border-radius:12px !important;
    font-weight:700 !important;
}

h2,h3{
    color:#172033;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data(show_spinner=False)
def load_default():
    return pd.read_csv("retail_store_inventory.csv")

@st.cache_data(show_spinner=False)
def load_uploaded(file_bytes):
    return pd.read_csv(BytesIO(file_bytes))

if st.session_state.get("uploaded_file_bytes") is not None:
    df = load_uploaded(st.session_state.get("uploaded_file_bytes"))
else:
    df = load_default()

if "Date" in df.columns:
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("## 📦 Smarty Stocks Pro")
    st.caption("Enterprise Inventory Intelligence")

    if st.button("Dashboard", use_container_width=True):
        st.session_state["page"] = "Dashboard"
        st.rerun()

    if st.button("Demand Forecasting", use_container_width=True):
        st.session_state["page"] = "Demand Forecasting"
        st.rerun()

    if st.button("Inventory Decision & Control", use_container_width=True):
        st.session_state["page"] = "Inventory Decision & Control"
        st.rerun()

    if st.button("Settings", use_container_width=True):
        st.session_state["page"] = "Settings"
        st.rerun()

page = st.session_state.get("page", "Dashboard")

# =====================================================
# HELPERS
# =====================================================
def safe_download_button(df_in: pd.DataFrame, filename: str, label: str):
    csv = df_in.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

def render_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.subheader("Filters")
    working = data.copy()

    c1, c2, c3, c4 = st.columns([1, 1, 1, 0.45])

    if "Store ID" in working.columns:
        store_options = sorted(working["Store ID"].astype(str).dropna().unique().tolist())
        if st.session_state.get("selected_stores") is None:
            st.session_state["selected_stores"] = store_options

        with c1:
            stores = st.multiselect(
                "Store",
                store_options,
                default=st.session_state.get("selected_stores", store_options)
            )
        st.session_state["selected_stores"] = stores
        if stores:
            working = working[working["Store ID"].astype(str).isin(stores)]

    if "Category" in working.columns:
        category_options = sorted(working["Category"].astype(str).dropna().unique().tolist())
        if st.session_state.get("selected_categories") is None:
            st.session_state["selected_categories"] = category_options

        with c2:
            categories = st.multiselect(
                "Category",
                category_options,
                default=st.session_state.get("selected_categories", category_options)
            )
        st.session_state["selected_categories"] = categories
        if categories:
            working = working[working["Category"].astype(str).isin(categories)]

    if "Date" in working.columns and working["Date"].notna().any():
        min_d = working["Date"].min().date()
        max_d = working["Date"].max().date()

        if st.session_state.get("date_range") is None:
            st.session_state["date_range"] = (min_d, max_d)

        with c3:
            dr = st.date_input(
                "Date Range",
                value=st.session_state.get("date_range", (min_d, max_d)),
                min_value=min_d,
                max_value=max_d
            )

        if isinstance(dr, tuple) and len(dr) == 2:
            st.session_state["date_range"] = dr
        elif isinstance(dr, list) and len(dr) == 2:
            st.session_state["date_range"] = tuple(dr)

        start, end = st.session_state.get("date_range", (min_d, max_d))
        working = working[
            (working["Date"] >= pd.to_datetime(start)) &
            (working["Date"] <= pd.to_datetime(end))
        ]

    with c4:
        st.write("")
        st.write("")
        if st.button("Reset Filters", use_container_width=True):
            st.session_state["selected_stores"] = None
            st.session_state["selected_categories"] = None
            st.session_state["date_range"] = None
            st.rerun()

    st.markdown("---")
    return working

def clip_limit(a, b=None, use_clip=True, percentile=95):
    vals = pd.to_numeric(a, errors="coerce").dropna().values
    if b is not None:
        vals2 = pd.to_numeric(b, errors="coerce").dropna().values
        if len(vals2):
            vals = np.concatenate([vals, vals2])
    if len(vals) == 0:
        return None
    if use_clip:
        return float(np.percentile(vals, percentile))
    return float(np.max(vals))

def make_dashboard_ai_explanation(df_in: pd.DataFrame, mae: float, r2: float) -> str:
    reorder = int((df_in["Recommended Action"] == "Reorder Inventory").sum())
    maintain = int((df_in["Recommended Action"] == "Maintain Level").sum())
    reduce = int((df_in["Recommended Action"] == "Reduce Stock").sum())
    anomalies = int((df_in["Anomaly Status"] == "Anomaly").sum())

    avg_actual = float(df_in["Actual Units Sold"].mean())
    avg_pred = float(df_in["Predicted Demand"].mean())
    avg_err = float(df_in["Absolute Error"].mean())
    avg_inventory = float(df_in["Inventory Level"].mean()) if "Inventory Level" in df_in.columns else 0.0

    top_store = df_in["Store ID"].astype(str).value_counts().idxmax() if "Store ID" in df_in.columns else "N/A"
    top_category = df_in["Category"].astype(str).value_counts().idxmax() if "Category" in df_in.columns else "N/A"

    forecast_direction = "under-forecasting" if avg_pred < avg_actual else "over-forecasting"

    return f"""
The dashboard is currently analysing **{len(df_in):,} rows** after applying the selected filters.

Average actual sales are **{avg_actual:.2f} units**, while average predicted demand is **{avg_pred:.2f} units**, which suggests the model is **{forecast_direction}** slightly in this filtered view.

Average inventory level is **{avg_inventory:.2f} units** and the average absolute forecast error is **{avg_err:.2f} units**.

The decision engine recommended:
- **{reorder}** reorder actions
- **{maintain}** maintain actions
- **{reduce}** reduce stock actions

The anomaly model flagged **{anomalies} unusual records** that deviate from normal sales behaviour.

The most represented store is **{top_store}**, while the most represented category is **{top_category}**.

Model quality in this filtered segment:
- **MAE:** {mae:.2f}
- **R²:** {r2:.2f}
""".strip()

def make_forecast_ai_explanation(df_in: pd.DataFrame, mae: float, r2: float) -> str:
    avg_actual = float(df_in["Actual Units Sold"].mean())
    avg_pred = float(df_in["Predicted Demand"].mean())
    avg_err = float(df_in["Absolute Error"].mean())
    max_err = float(df_in["Absolute Error"].max())

    return f"""
This forecast results table shows the Random Forest model output for the currently filtered data.

Each row compares:
- **Actual Units Sold**: the true observed sales
- **Predicted Demand**: the model's estimated demand
- **Residual**: the difference between actual and predicted values
- **Absolute Error**: the size of the prediction error regardless of direction

In this filtered view, average actual sales are **{avg_actual:.2f} units** and average predicted demand is **{avg_pred:.2f} units**.

The average absolute error is **{avg_err:.2f} units**, while the largest observed forecast error is **{max_err:.2f} units**.

This means the table helps you identify where the model is predicting well and where it is struggling, which is useful for evaluating forecast reliability before inventory decisions are made.

Model quality:
- **MAE:** {mae:.2f}
- **R²:** {r2:.2f}
""".strip()

def make_anomaly_ai_explanation(df_in: pd.DataFrame) -> str:
    if df_in.empty:
        return "No anomalies were detected for the current filters."

    avg_score = float(df_in["Anomaly Score"].mean()) if "Anomaly Score" in df_in.columns else 0.0
    max_score = float(df_in["Anomaly Score"].max()) if "Anomaly Score" in df_in.columns else 0.0

    top_store = df_in["Store ID"].astype(str).value_counts().idxmax() if "Store ID" in df_in.columns else "N/A"
    top_category = df_in["Category"].astype(str).value_counts().idxmax() if "Category" in df_in.columns else "N/A"

    return f"""
This anomaly monitoring table shows only the rows classified as unusual by the Isolation Forest model.

An anomaly means the row behaves differently from the normal pattern learned from:
- actual sales
- predicted demand
- forecast residuals

In the current filtered view:
- **{len(df_in):,} anomalies** were detected
- Average anomaly score is **{avg_score:.3f}**
- Highest anomaly score is **{max_score:.3f}**

Higher anomaly scores indicate more unusual behaviour.

The store most frequently appearing in anomalous rows is **{top_store}**, and the category most frequently appearing is **{top_category}**.

This table is useful for identifying sudden demand spikes, irregular forecast behaviour, possible stock issues, or unusual sales patterns that may require investigation.
""".strip()

def make_decision_ai_explanation(df_in: pd.DataFrame) -> str:
    reorder = int((df_in["Recommended Action"] == "Reorder Inventory").sum())
    maintain = int((df_in["Recommended Action"] == "Maintain Level").sum())
    reduce = int((df_in["Recommended Action"] == "Reduce Stock").sum())
    anomalies = int((df_in["Anomaly Status"] == "Anomaly").sum())

    avg_action = float(df_in["Action Score"].mean()) if "Action Score" in df_in.columns else 0.0
    avg_inventory = float(df_in["Inventory Level"].mean()) if "Inventory Level" in df_in.columns else 0.0
    avg_pred = float(df_in["Predicted Demand"].mean()) if "Predicted Demand" in df_in.columns else 0.0

    return f"""
This decision report combines forecast demand with current inventory levels using fuzzy logic.

The fuzzy logic system converts numeric values into business-style decisions:
- **Reorder Inventory** when demand appears high relative to stock
- **Maintain Level** when stock and demand are reasonably balanced
- **Reduce Stock** when inventory appears high relative to expected demand

In the current filtered view:
- **{reorder}** rows were flagged for reorder
- **{maintain}** rows were marked as maintain
- **{reduce}** rows were marked as reduce stock
- **{anomalies}** rows were also flagged as anomalies

Average predicted demand is **{avg_pred:.2f} units**, average inventory level is **{avg_inventory:.2f} units**, and the average action score is **{avg_action:.2f}**.

This table helps turn AI predictions into operational decisions, making it easier for managers to prioritise replenishment, avoid overstocking, and review unusual situations.
""".strip()

# =====================================================
# PIPELINE
# =====================================================
@st.cache_data(show_spinner=False)
def run_forecast(data: pd.DataFrame, trees: int, test_ratio: float):
    features = ["Inventory Level", "Units Ordered"]
    model_df = data.dropna(subset=features + ["Units Sold"]).copy()

    if len(model_df) < 20:
        return pd.DataFrame(), 0.0, 0.0

    X = model_df[features]
    y = model_df["Units Sold"]

    split = int(len(model_df) * (1 - test_ratio))
    split = max(1, min(split, len(model_df) - 1))

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = RandomForestRegressor(
        n_estimators=trees,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    out = model_df.iloc[split:].copy()
    out["Actual Units Sold"] = y_test.values
    out["Predicted Demand"] = pred
    out["Residual"] = out["Actual Units Sold"] - out["Predicted Demand"]
    out["Absolute Error"] = np.abs(out["Residual"])

    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    return out, float(mae), float(r2)

@st.cache_data(show_spinner=False)
def fuzzy_logic(data: pd.DataFrame):
    df_fuzzy = data.copy()

    demand = ctrl.Antecedent(np.arange(0, 501, 1), "demand")
    inventory = ctrl.Antecedent(np.arange(0, 501, 1), "inventory")
    action = ctrl.Consequent(np.arange(0, 101, 1), "action")

    demand["low"] = fuzz.trimf(demand.universe, [0, 0, 200])
    demand["medium"] = fuzz.trimf(demand.universe, [100, 250, 400])
    demand["high"] = fuzz.trimf(demand.universe, [300, 500, 500])

    inventory["low"] = fuzz.trimf(inventory.universe, [0, 0, 200])
    inventory["medium"] = fuzz.trimf(inventory.universe, [100, 250, 400])
    inventory["high"] = fuzz.trimf(inventory.universe, [300, 500, 500])

    action["reduce"] = fuzz.trimf(action.universe, [0, 0, 40])
    action["maintain"] = fuzz.trimf(action.universe, [30, 50, 70])
    action["reorder"] = fuzz.trimf(action.universe, [60, 100, 100])

    rules = [
        ctrl.Rule(demand["high"] & inventory["low"], action["reorder"]),
        ctrl.Rule(demand["high"] & inventory["medium"], action["reorder"]),
        ctrl.Rule(demand["high"] & inventory["high"], action["maintain"]),
        ctrl.Rule(demand["medium"] & inventory["low"], action["reorder"]),
        ctrl.Rule(demand["medium"] & inventory["medium"], action["maintain"]),
        ctrl.Rule(demand["medium"] & inventory["high"], action["reduce"]),
        ctrl.Rule(demand["low"] & inventory["low"], action["maintain"]),
        ctrl.Rule(demand["low"] & inventory["medium"], action["reduce"]),
        ctrl.Rule(demand["low"] & inventory["high"], action["reduce"])
    ]

    system = ctrl.ControlSystem(rules)

    actions = []
    scores = []
    max_rows = min(len(df_fuzzy), 4000)

    for _, row in df_fuzzy.head(max_rows).iterrows():
        sim = ctrl.ControlSystemSimulation(system)
        sim.input["demand"] = float(np.clip(row["Predicted Demand"], 0, 500))
        sim.input["inventory"] = float(np.clip(row["Inventory Level"], 0, 500))
        sim.compute()

        score = sim.output["action"] if "action" in sim.output else 50.0

        if score < 33:
            act = "Reduce Stock"
        elif score < 66:
            act = "Maintain Level"
        else:
            act = "Reorder Inventory"

        actions.append(act)
        scores.append(score)

    df_fuzzy = df_fuzzy.head(max_rows).copy()
    df_fuzzy["Recommended Action"] = actions
    df_fuzzy["Action Score"] = scores

    return df_fuzzy

@st.cache_data(show_spinner=False)
def detect_anomaly(data: pd.DataFrame, contamination: float):
    df_anom = data.copy()
    features = df_anom[["Actual Units Sold", "Predicted Demand", "Residual"]]

    model = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    labels = model.fit_predict(features)
    raw_scores = -model.decision_function(features)

    if raw_scores.max() - raw_scores.min() > 1e-9:
        norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
    else:
        norm_scores = np.zeros_like(raw_scores)

    df_anom["Anomaly Status"] = ["Anomaly" if x == -1 else "Normal" for x in labels]
    df_anom["Anomaly Score"] = np.round(norm_scores, 3)

    return df_anom

def run_pipeline(filtered: pd.DataFrame):
    with st.spinner("Running analysis..."):
        forecast_df, mae, r2 = run_forecast(
            filtered,
            int(st.session_state.get("trees", 120)),
            float(st.session_state.get("test_ratio", 0.20))
        )

        if forecast_df.empty:
            return forecast_df, mae, r2

        fuzzy_df = fuzzy_logic(forecast_df)
        final_df = detect_anomaly(fuzzy_df, float(st.session_state.get("anomaly", 0.05)))

    return final_df, mae, r2

# =====================================================
# COMMON HEADER
# =====================================================
if page != "Settings":
    title_col, summary_col = st.columns([3, 1])

    with title_col:
        st.markdown('<div class="main-title">Smarty Stocks Pro</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Forecast demand, optimise stock decisions, and monitor anomalies across your retail network.</div>', unsafe_allow_html=True)

    with summary_col:
        dataset_name = st.session_state.get("uploaded_file_name") or "Default CSV"
        st.markdown(
            f"""
            <div class="summary-box">
                <b>Dataset:</b> {dataset_name}<br>
                <b>Trees:</b> {st.session_state.get("trees", 120)}<br>
                <b>Test Ratio:</b> {int(st.session_state.get("test_ratio", 0.20) * 100)}%<br>
                <b>Anomaly:</b> {st.session_state.get("anomaly", 0.05):.2f}
            </div>
            """,
            unsafe_allow_html=True
        )

# =====================================================
# DASHBOARD
# =====================================================
if page == "Dashboard":
    st.markdown('<div class="section-chip">Dashboard</div>', unsafe_allow_html=True)

    filtered = render_filters(df)
    results, mae, r2 = run_pipeline(filtered)

    if results.empty:
        st.warning("Not enough data for the current filters.")
        st.stop()

    anomalies = int((results["Anomaly Status"] == "Anomaly").sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card"><div class="kpi-label">Rows</div><div class="kpi-value">{len(results):,}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card"><div class="kpi-label">MAE</div><div class="kpi-value">{mae:.2f}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card"><div class="kpi-label">R²</div><div class="kpi-value">{r2:.2f}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card"><div class="kpi-label">Anomalies</div><div class="kpi-value">{anomalies}</div></div>', unsafe_allow_html=True)

    st.markdown("### AI Explanation")
    st.markdown(f'<div class="info-box">{make_dashboard_ai_explanation(results, mae, r2)}</div>', unsafe_allow_html=True)

    st.markdown("### Visual Analytics")
    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Actual vs Predicted Demand")
        fig = plt.figure(figsize=(7, 4))
        plot_df = results.copy()
        lim = clip_limit(
            results["Actual Units Sold"],
            results["Predicted Demand"],
            use_clip=bool(st.session_state.get("clip_charts", True)),
            percentile=int(st.session_state.get("clip_percentile", 95))
        )
        if lim is not None and bool(st.session_state.get("clip_charts", True)):
            plot_df = plot_df[
                (plot_df["Actual Units Sold"] <= lim) &
                (plot_df["Predicted Demand"] <= lim)
            ]
        plt.scatter(plot_df["Actual Units Sold"], plot_df["Predicted Demand"], alpha=0.25)
        plt.plot([0, 400], [0, 400], "--")
        plt.xlim(0, 350)
        plt.ylim(0, 400)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row1_col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Recommended Action Distribution")
        fig2 = plt.figure(figsize=(7, 4))
        results["Recommended Action"].value_counts().plot(kind="bar")
        plt.ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Top Anomalies")
    top_anom = results[results["Anomaly Status"] == "Anomaly"].sort_values("Anomaly Score", ascending=False)
    st.markdown(f'<div class="info-box">{make_anomaly_ai_explanation(top_anom)}</div>', unsafe_allow_html=True)
    st.dataframe(top_anom.head(25), use_container_width=True, height=320)

    st.markdown("### Executive Report")
    st.dataframe(results, use_container_width=True, height=520)
    safe_download_button(results, "executive_report.csv", "Download Executive Report")

# =====================================================
# DEMAND FORECASTING
# =====================================================
elif page == "Demand Forecasting":
    st.markdown('<div class="section-chip">Demand Forecasting</div>', unsafe_allow_html=True)

    filtered = render_filters(df)
    results, mae, r2 = run_forecast(
        filtered,
        int(st.session_state.get("trees", 120)),
        float(st.session_state.get("test_ratio", 0.20))
    )

    if results.empty:
        st.warning("Not enough data for the current filters.")
        st.stop()

    c1, c2 = st.columns(2)
    c1.metric("MAE", round(mae, 2))
    c2.metric("R²", round(r2, 2))

    st.markdown("### AI Explanation")
    st.markdown(f'<div class="info-box">{make_forecast_ai_explanation(results, mae, r2)}</div>', unsafe_allow_html=True)

    f1, f2 = st.columns(2)
    sample = results.head(200)

    with f1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Forecast Trend")
        fig = plt.figure(figsize=(7, 4))
        plt.plot(sample["Actual Units Sold"].values, label="Actual")
        plt.plot(sample["Predicted Demand"].values, label="Predicted")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with f2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Absolute Error Trend")
        fig2 = plt.figure(figsize=(7, 4))
        plt.plot(sample["Absolute Error"].values)
        plt.ylabel("Absolute Error")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Forecast Results")
    st.dataframe(results, use_container_width=True, height=520)
    safe_download_button(results, "forecast_results.csv", "Download Forecast Results")

# =====================================================
# INVENTORY DECISION & CONTROL
# =====================================================
elif page == "Inventory Decision & Control":
    st.markdown('<div class="section-chip">Inventory Decision & Control</div>', unsafe_allow_html=True)

    filtered = render_filters(df)
    results, mae, r2 = run_pipeline(filtered)

    if results.empty:
        st.warning("Not enough data for the current filters.")
        st.stop()

    st.markdown("### AI Explanation")
    st.markdown(f'<div class="info-box">{make_decision_ai_explanation(results)}</div>', unsafe_allow_html=True)

    d1, d2 = st.columns(2)

    with d1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Action Score Distribution")
        fig = plt.figure(figsize=(7, 4))
        grouped = results.groupby("Recommended Action")["Action Score"]
        data_box = []
        labels_box = []
        for name, group in grouped:
            if len(group) > 0:
                data_box.append(group.values)
                labels_box.append(name)
        if len(data_box) > 0:
            plt.boxplot(data_box, tick_labels=labels_box)
            plt.ylabel("Action Score")
        else:
            plt.text(0.5, 0.5, "No data available", ha="center")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with d2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Anomaly Status Count")
        fig2 = plt.figure(figsize=(7, 4))
        results["Anomaly Status"].value_counts().plot(kind="bar")
        plt.ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Anomaly Monitoring")
    anomaly_table = results[results["Anomaly Status"] == "Anomaly"].sort_values("Anomaly Score", ascending=False)
    st.markdown(f'<div class="info-box">{make_anomaly_ai_explanation(anomaly_table)}</div>', unsafe_allow_html=True)
    st.dataframe(anomaly_table, use_container_width=True, height=360)
    safe_download_button(anomaly_table, "anomaly_results.csv", "Download Anomaly Results")

    st.markdown("### Decision Report")
    st.markdown(f'<div class="info-box">{make_decision_ai_explanation(results)}</div>', unsafe_allow_html=True)
    st.dataframe(results, use_container_width=True, height=520)
    safe_download_button(results, "decision_results.csv", "Download Decision Report")

# =====================================================
# SETTINGS
# =====================================================
elif page == "Settings":
    st.markdown('<div class="main-title">Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Adjust dataset source and model behaviour.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        st.session_state["uploaded_file_bytes"] = uploaded.getvalue()
        st.session_state["uploaded_file_name"] = uploaded.name
        st.success(f"Loaded file: {uploaded.name}")

    if st.session_state.get("uploaded_file_name") is not None:
        st.info(f"Current dataset: {st.session_state.get('uploaded_file_name')}")
        if st.button("Use Default Dataset Again"):
            st.session_state["uploaded_file_bytes"] = None
            st.session_state["uploaded_file_name"] = None
            st.rerun()

    st.slider("Random Forest Trees", 50, 300, key="trees")
    st.slider("Test Ratio", 0.10, 0.40, key="test_ratio")
    st.slider("Anomaly Sensitivity", 0.01, 0.10, key="anomaly")
    st.checkbox("Use Outlier Safe Charts", key="clip_charts")
    st.slider("Chart Percentile", 85, 100, key="clip_percentile")
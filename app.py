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

.block-container{
    padding-top:1.2rem;
    padding-bottom:2rem;
}

/* Sidebar */
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
    margin-bottom:0.45rem !important;
}

section[data-testid="stSidebar"] .stButton > button:hover{
    background:rgba(255,255,255,0.16) !important;
}

/* Typography */
.main-title{
    font-size:2.25rem;
    font-weight:800;
    color:#172033;
    margin-bottom:0.12rem;
}

.sub-title{
    color:#6b7280;
    font-size:0.97rem;
    margin-bottom:1rem;
}

.section-title{
    font-size:1.55rem;
    font-weight:800;
    color:#172033;
    margin:0.2rem 0 0.8rem 0;
}

.section-chip{
    display:inline-block;
    background:#2563eb;
    color:white;
    padding:0.45rem 0.95rem;
    border-radius:999px;
    font-weight:700;
    font-size:0.84rem;
    margin-bottom:0.65rem;
}

/* Cards */
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
    height:100%;
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

/* Gauge */
.gauge-wrap{
    width:100%;
    background:#e5e7eb;
    border-radius:999px;
    height:14px;
    overflow:hidden;
    margin-top:10px;
}

.gauge-bar{
    height:14px;
    border-radius:999px;
}

.small-note{
    color:#6b7280;
    font-size:0.88rem;
    margin-top:10px;
}

/* Misc */
.stDownloadButton > button{
    border-radius:12px !important;
    font-weight:700 !important;
}

div[data-testid="stDataFrame"]{
    border-radius:14px;
    overflow:hidden;
    border:1px solid #e7ecf5;
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

def existing_cols(df_in: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df_in.columns]

def render_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.markdown('<div class="section-title">Filters</div>', unsafe_allow_html=True)
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

def accuracy_label(r2_value: float):
    if r2_value < 0.30:
        return "Low", "#dc2626"
    if r2_value < 0.60:
        return "Moderate", "#d97706"
    return "Strong", "#16a34a"

def render_accuracy_gauge(r2_value: float):
    pct = max(0, min(100, r2_value * 100))
    label, color = accuracy_label(r2_value)
    st.markdown(
        f"""
        <div class="panel">
            <h3 style="margin-bottom:0.35rem;">Forecast Accuracy Gauge</h3>
            <div style="font-size:1.9rem;font-weight:800;color:#172033;">{pct:.1f}%</div>
            <div class="gauge-wrap">
                <div class="gauge-bar" style="width:{pct:.1f}%; background:{color};"></div>
            </div>
            <div class="small-note">
                Accuracy interpretation: <b style="color:{color};">{label}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def style_status_dataframe(df_in: pd.DataFrame):
    def color_action(val):
        if val == "Reorder Inventory":
            return "background-color: #fee2e2; color: #991b1b; font-weight: 700;"
        if val == "Maintain Level":
            return "background-color: #dbeafe; color: #1d4ed8; font-weight: 700;"
        if val == "Reduce Stock":
            return "background-color: #dcfce7; color: #166534; font-weight: 700;"
        return ""

    def color_anomaly(val):
        if val == "Anomaly":
            return "background-color: #fef3c7; color: #92400e; font-weight: 700;"
        if val == "Normal":
            return "background-color: #f3f4f6; color: #374151; font-weight: 700;"
        return ""

    styler = df_in.style
    if "Recommended Action" in df_in.columns:
        styler = styler.map(color_action, subset=["Recommended Action"])
    if "Anomaly Status" in df_in.columns:
        styler = styler.map(color_anomaly, subset=["Anomaly Status"])

    format_dict = {}
    for col in ["Predicted Demand", "Residual", "Absolute Error", "Action Score", "Anomaly Score"]:
        if col in df_in.columns:
            format_dict[col] = "{:.2f}"
    if format_dict:
        styler = styler.format(format_dict)

    return styler

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
The dashboard is currently analysing **{len(df_in):,} evaluated rows** after the selected filters were applied.

Average actual sales are **{avg_actual:.2f} units**, while average predicted demand is **{avg_pred:.2f} units**, suggesting the model is **{forecast_direction}** slightly in this view.

Average inventory level is **{avg_inventory:.2f} units** and the average absolute forecast error is **{avg_err:.2f} units**.

The decision engine recommended:
- **{reorder}** reorder actions
- **{maintain}** maintain actions
- **{reduce}** reduce stock actions

The anomaly model flagged **{anomalies} unusual records**.

The most represented store is **{top_store}**, while the most represented category is **{top_category}**.

Model quality:
- **MAE:** {mae:.2f}
- **R²:** {r2:.2f}
""".strip()

def make_management_insights(df_in: pd.DataFrame) -> str:
    if df_in.empty:
        return "No insights are available for the current filters."

    sales_by_store = (
        df_in.groupby("Store ID")["Actual Units Sold"].sum().sort_values(ascending=False)
        if "Store ID" in df_in.columns else pd.Series(dtype=float)
    )

    best_store = sales_by_store.index[0] if len(sales_by_store) > 0 else "N/A"
    worst_store = sales_by_store.index[-1] if len(sales_by_store) > 0 else "N/A"

    anomaly_count = int((df_in["Anomaly Status"] == "Anomaly").sum())
    top_category = df_in["Category"].astype(str).value_counts().idxmax() if "Category" in df_in.columns else "N/A"
    dominant_action = df_in["Recommended Action"].astype(str).value_counts().idxmax() if "Recommended Action" in df_in.columns else "N/A"

    return f"""
Management insights for the current filtered view:

- **Best performing store:** {best_store}
- **Weakest performing store:** {worst_store}
- **Most active category:** {top_category}
- **Most common inventory decision:** {dominant_action}
- **Anomalies detected:** {anomaly_count}

This suggests that managers should prioritise reviewing anomalous records while focusing replenishment or stock policy decisions on the most active store-category combinations.
""".strip()

def make_forecast_ai_explanation(df_in: pd.DataFrame, mae: float, r2: float) -> str:
    avg_actual = float(df_in["Actual Units Sold"].mean())
    avg_pred = float(df_in["Predicted Demand"].mean())
    avg_err = float(df_in["Absolute Error"].mean())
    max_err = float(df_in["Absolute Error"].max())

    return f"""
This forecast results table shows the Random Forest model output for the currently filtered data.

Each row compares:
- **Actual Units Sold**
- **Predicted Demand**
- **Residual**
- **Absolute Error**

Average actual sales are **{avg_actual:.2f} units**, while average predicted demand is **{avg_pred:.2f} units**.

The average absolute error is **{avg_err:.2f} units**, while the largest observed forecast error is **{max_err:.2f} units**.

This view helps identify where the forecasting model is performing well and where reliability becomes weaker.

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

In the current filtered view:
- **{len(df_in):,} anomalies** were detected
- Average anomaly score is **{avg_score:.3f}**
- Highest anomaly score is **{max_score:.3f}**

The most frequently affected store is **{top_store}**, while the most affected category is **{top_category}**.

These records may indicate unusual demand spikes, irregular stock behaviour, exceptional transactions, or operational inconsistencies that warrant further review.
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

The fuzzy logic system converts numeric values into business-oriented actions:
- **Reorder Inventory**
- **Maintain Level**
- **Reduce Stock**

In the current filtered view:
- **{reorder}** rows were flagged for reorder
- **{maintain}** rows were marked as maintain
- **{reduce}** rows were marked as reduce stock
- **{anomalies}** rows were also flagged as anomalies

Average predicted demand is **{avg_pred:.2f} units**, average inventory level is **{avg_inventory:.2f} units**, and the average action score is **{avg_action:.2f}**.

This view translates AI outputs into operational inventory decisions.
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
# HEADER
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

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f'<div class="kpi-card"><div class="kpi-label">Rows</div><div class="kpi-value">{len(results):,}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="kpi-label">MAE</div><div class="kpi-value">{mae:.2f}</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><div class="kpi-label">R²</div><div class="kpi-value">{r2:.2f}</div></div>', unsafe_allow_html=True)
    k4.markdown(f'<div class="kpi-card"><div class="kpi-label">Anomalies</div><div class="kpi-value">{anomalies}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    ai_col, gauge_col = st.columns([1.45, 1])

    with ai_col:
        st.markdown('<div class="section-title">AI Explanation</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="info-box">{make_dashboard_ai_explanation(results, mae, r2)}</div>', unsafe_allow_html=True)

    with gauge_col:
        st.markdown('<div class="section-title">Forecast Accuracy</div>', unsafe_allow_html=True)
        render_accuracy_gauge(r2)

    st.markdown("")
    st.markdown('<div class="section-title">Management Insights</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="info-box">{make_management_insights(results)}</div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="section-title">Visual Analytics</div>', unsafe_allow_html=True)

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Demand Trend Over Time")
        fig = plt.figure(figsize=(7, 4))
        if "Date" in results.columns and results["Date"].notna().any():
            trend_df = (
                results.dropna(subset=["Date"])
                .sort_values("Date")
                .groupby("Date")[["Actual Units Sold", "Predicted Demand"]]
                .mean()
                .reset_index()
            ).tail(120)
            plt.plot(trend_df["Date"], trend_df["Actual Units Sold"], label="Actual")
            plt.plot(trend_df["Date"], trend_df["Predicted Demand"], label="Predicted")
            plt.legend()
            plt.xticks(rotation=30)
        else:
            plt.text(0.5, 0.5, "Date column unavailable", ha="center")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row1_col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Actual vs Predicted Demand")
        fig2 = plt.figure(figsize=(7, 4))
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
        plt.scatter(plot_df["Actual Units Sold"], plot_df["Predicted Demand"], alpha=0.22)
        plt.plot([0, 500], [0, 500], "--")
        plt.xlim(0, 500)
        plt.ylim(0, 500)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Recommended Action Distribution")
        fig3 = plt.figure(figsize=(7, 4))
        results["Recommended Action"].value_counts().plot(kind="bar")
        plt.ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row2_col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Residual Distribution")
        fig4 = plt.figure(figsize=(7, 4))
        plt.hist(results["Residual"], bins=40)
        plt.xlabel("Residual")
        plt.ylabel("Frequency")
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Store Performance Ranking")
        ranking = results.groupby("Store ID").agg(
            Total_Sales=("Actual Units Sold", "sum"),
            Avg_Error=("Absolute Error", "mean"),
            Anomalies=("Anomaly Status", lambda x: (x == "Anomaly").sum())
        ).reset_index()
        ranking["Risk Score"] = ranking["Avg_Error"] + ranking["Anomalies"] * 2
        ranking = ranking.sort_values(["Total_Sales", "Risk Score"], ascending=[False, True]).reset_index(drop=True)
        ranking.index = ranking.index + 1
        st.dataframe(ranking[existing_cols(ranking, ["Store ID", "Total_Sales", "Avg_Error", "Anomalies", "Risk Score"])], use_container_width=True, height=320)
        st.markdown('</div>', unsafe_allow_html=True)

    with row3_col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Anomaly Heatmap (Store x Category)")
        fig5 = plt.figure(figsize=(7, 4))
        heat_df = results.copy()
        heat_df["Anomaly Binary"] = (heat_df["Anomaly Status"] == "Anomaly").astype(int)
        if "Store ID" in heat_df.columns and "Category" in heat_df.columns:
            pivot = heat_df.pivot_table(
                index="Store ID",
                columns="Category",
                values="Anomaly Binary",
                aggfunc="sum",
                fill_value=0
            )
            if pivot.shape[0] > 0 and pivot.shape[1] > 0:
                plt.imshow(pivot.values, aspect="auto")
                plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
                plt.yticks(range(len(pivot.index)), pivot.index)
                plt.colorbar(label="Anomaly Count")
            else:
                plt.text(0.5, 0.5, "No heatmap data", ha="center")
        else:
            plt.text(0.5, 0.5, "Required columns unavailable", ha="center")
        plt.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")
    bottom1, bottom2 = st.columns([1, 1])

    with bottom1:
        st.markdown('<div class="section-title">Top Anomalies</div>', unsafe_allow_html=True)
        top_anom = results[results["Anomaly Status"] == "Anomaly"].sort_values("Anomaly Score", ascending=False)
        st.markdown(f'<div class="info-box">{make_anomaly_ai_explanation(top_anom)}</div>', unsafe_allow_html=True)
        top_anom_cols = existing_cols(top_anom, [
            "Date", "Store ID", "Category", "Inventory Level",
            "Actual Units Sold", "Predicted Demand", "Residual",
            "Anomaly Status", "Anomaly Score"
        ])
        st.dataframe(style_status_dataframe(top_anom[top_anom_cols].head(25)), use_container_width=True, height=320)

    with bottom2:
        st.markdown('<div class="section-title">Executive Report Preview</div>', unsafe_allow_html=True)
        preview_cols = existing_cols(results, [
            "Date", "Store ID", "Category", "Inventory Level",
            "Actual Units Sold", "Predicted Demand",
            "Recommended Action", "Anomaly Status", "Anomaly Score"
        ])
        st.dataframe(style_status_dataframe(results[preview_cols].head(25)), use_container_width=True, height=320)

    st.markdown("")
    st.markdown('<div class="section-title">Executive Report</div>', unsafe_allow_html=True)
    executive_cols = existing_cols(results, [
        "Date", "Store ID", "Category", "Inventory Level", "Units Ordered",
        "Actual Units Sold", "Predicted Demand", "Residual", "Absolute Error",
        "Recommended Action", "Action Score", "Anomaly Status", "Anomaly Score"
    ])
    st.dataframe(style_status_dataframe(results[executive_cols]), use_container_width=True, height=520)
    safe_download_button(results[executive_cols], "executive_report.csv", "Download Executive Report")

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

    st.markdown('<div class="section-title">AI Explanation</div>', unsafe_allow_html=True)
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

    st.markdown("")
    st.markdown('<div class="section-title">Forecast Results</div>', unsafe_allow_html=True)
    forecast_cols = existing_cols(results, [
        "Date", "Store ID", "Category", "Inventory Level", "Units Ordered",
        "Actual Units Sold", "Predicted Demand", "Residual", "Absolute Error"
    ])
    st.dataframe(style_status_dataframe(results[forecast_cols]), use_container_width=True, height=520)
    safe_download_button(results[forecast_cols], "forecast_results.csv", "Download Forecast Results")

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

    st.markdown('<div class="section-title">AI Explanation</div>', unsafe_allow_html=True)
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

    st.markdown("")
    st.markdown('<div class="section-title">Anomaly Monitoring</div>', unsafe_allow_html=True)
    anomaly_table = results[results["Anomaly Status"] == "Anomaly"].sort_values("Anomaly Score", ascending=False)
    st.markdown(f'<div class="info-box">{make_anomaly_ai_explanation(anomaly_table)}</div>', unsafe_allow_html=True)
    anomaly_cols = existing_cols(anomaly_table, [
        "Date", "Store ID", "Category", "Inventory Level",
        "Actual Units Sold", "Predicted Demand", "Residual",
        "Anomaly Status", "Anomaly Score", "Recommended Action"
    ])
    st.dataframe(style_status_dataframe(anomaly_table[anomaly_cols]), use_container_width=True, height=360)
    safe_download_button(anomaly_table[anomaly_cols], "anomaly_results.csv", "Download Anomaly Results")

    st.markdown("")
    st.markdown('<div class="section-title">Decision Report</div>', unsafe_allow_html=True)
    decision_cols = existing_cols(results, [
        "Date", "Store ID", "Category", "Inventory Level",
        "Predicted Demand", "Recommended Action", "Action Score",
        "Anomaly Status", "Anomaly Score"
    ])
    st.dataframe(style_status_dataframe(results[decision_cols]), use_container_width=True, height=520)
    safe_download_button(results[decision_cols], "decision_results.csv", "Download Decision Report")

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

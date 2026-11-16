import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Driven Trade & Data Analysis System",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>

/* =========================
   Remove Top Space
========================= */

.block-container{
    padding-top: 180px !important;
}

/* =========================
   Floating Hero Header
========================= */

.hero-header{

    position: fixed;

    top: 50px;
    left: 250px;
    right: 20px;

    min-height: 100px;

    display: flex;
    flex-direction: column;
    justify-content: center;

    z-index: 9999;

    padding: 20px 40px;

    border-radius: 18px;

    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e40af 40%,
        #2563eb 70%,
        #38bdf8 100%
    );

    color: white;

    border: 1px solid rgba(255,255,255,0.15);

    backdrop-filter: blur(18px);

    box-shadow:
        0 12px 35px rgba(37,99,235,0.25),
        0 6px 15px rgba(0,0,0,0.15);

    transition: all .3s ease;
}

.hero-header:hover{

    transform: translateY(-2px);

    box-shadow:
        0 18px 40px rgba(37,99,235,.35);
}

/* Title */

.hero-header h1{

    margin: 0;

    padding: 0;

    font-size: 34px;

    font-weight: 800;

    line-height: 1.25;

    letter-spacing: 0.5px;
}

/* Subtitle */

.hero-header p{

    margin-top: 8px;

    margin-bottom: 0;

    font-size: 15px;

    opacity: .92;

    letter-spacing: .3px;
}

/* Badge */

.hero-badge{

    display: inline-block;

    width: fit-content;

    margin-top: 12px;

    padding: 6px 14px;

    border-radius: 50px;

    background: rgba(255,255,255,.15);

    border: 1px solid rgba(255,255,255,.20);

    font-size: 12px;

    font-weight: 600;
}

/* =========================
   Sticky Tabs
========================= */

.stTabs{

    position: sticky;

    top: 145px;

    z-index: 999;

    background: rgba(255,255,255,.95);

    backdrop-filter: blur(10px);

    border-radius: 12px;

    padding: 8px;
}

/* =========================
   Tablet
========================= */

@media (max-width:1024px){

    .hero-header{

        left:20px;
        right:20px;

        padding:20px;
    }

    .hero-header h1{
        font-size:28px;
    }

    .block-container{
        padding-top:180px !important;
    }
}

/* =========================
   Mobile
========================= */

@media (max-width:768px){

    .hero-header{

        left:10px;
        right:10px;

        top:5px;

        min-height:130px;

        padding:18px;

        text-align:center;

        align-items:center;
    }

    .hero-header h1{

        font-size:22px;

        line-height:1.3;
    }

    .hero-header p{

        font-size:12px;
    }

    .block-container{
        padding-top:210px !important;
    }

    .stTabs{
        top:155px;
    }
}


            
/* ===================================
   Intelligence KPI Glass Cards
=================================== */

.insight-kpi{

    background: rgba(255,255,255,0.10);

    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);

    border: 1px solid rgba(255,255,255,0.25);

    border-radius: 16px;

    padding: 16px;

    text-align: center;

    min-height: 100px;

    transition: all .3s ease;

    box-shadow:
        0 8px 24px rgba(0,0,0,.08);
}

.insight-kpi:hover{

    transform: translateY(-5px);

    background: rgba(37,99,235,0.18);

    box-shadow:
        0 12px 30px rgba(74, 221, 130,.20);
}

.insight-kpi-icon{

    font-size:18px;

    margin-bottom:6px;
}

.insight-kpi-title{

    color:#64748b;

    font-size:10px;

    font-weight:700;

    text-transform:uppercase;

    letter-spacing:.5px;
}

.insight-kpi-value{

    margin-top:10px;

    font-size:21px;

    font-weight:800;

    color:#0f172a;
}

/* Dashboard Header */

.summary-header{

    background:linear-gradient(
        135deg,
        #1e3a8a,
        #2563eb
    );

    color:white;

    padding:16px 21px;

    border-radius:16px;

    margin-bottom:10px;

    font-size:21px;

    font-weight:700;

    box-shadow:
        0 10px 30px rgba(37,99,235,.25);
}

/* Tab Container */

.stTabs [data-baseweb="tab-list"]{
    gap: 12px;
}

/* Individual Tab */

.stTabs [data-baseweb="tab"]{

    height: 45px;

    background: #4ADD82;

    border-radius: 12px;

    padding-left: 20px;
    padding-right: 20px;

    font-weight: 600;

    border: 1px solid #e5e7eb;

    box-shadow: 0px 2px 8px rgba(0,0,0,.05);
}

/* Selected Tab */

.stTabs [aria-selected="true"]{

    background: linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    ) !important;

    color: white !important;

    border: none;

    box-shadow: 0px 8px 20px rgba(37,99,235,.25);
}

         
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-header">

<h3 style="font-size:30px; color:#e0e7ff; font-weight:700">
📊 AI-Enhanced Analysis and Forecasting for National Export and Import Trade
</h3>

<p style="font-size:15px;margin-top:2px; color:#e0e7ff; font-weight:600; font-style:italic">
AI Forecasting • Trade Intelligence • Advanced Analytics • Executive Dashboard
</p>

<div class="hero-badge">
🚀 Enterprise Analytics Platform
</div>

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions & Robust Caching
# ---------------------------------------------------------
@st.cache_data
def load_raw_and_clean_data(uploaded_file):
    """Load both Raw and Clean versions of the dataset"""
    df_raw_orig = pd.read_csv(uploaded_file)
    df_clean = df_raw_orig.copy()
    df_clean.columns = df_clean.columns.str.lower().str.strip().str.replace(' ', '_')
    
    unwanted_cols = [
        c for c in df_clean.columns 
        if 'unnamed' in c or (c == 'index' and df_clean[c].is_unique and df_clean[c].dtype in ['int64', 'float64'])
    ]
    if unwanted_cols:
        df_clean = df_clean.drop(columns=unwanted_cols)
        
    return df_raw_orig, df_clean

@st.cache_data
def process_trade_data(df):
    """Automated Feature Engineering for Trade Balance and Total Trade"""
    processed_df = df.copy().fillna(0)
    col_names_lower = [c.lower() for c in processed_df.columns]
    if 'export' in col_names_lower and 'import' in col_names_lower:
        exp_col = processed_df.columns[col_names_lower.index('export')]
        imp_col = processed_df.columns[col_names_lower.index('import')]
        processed_df['total_trade'] = processed_df[exp_col] + processed_df[imp_col]
        processed_df['trade_balance'] = processed_df[exp_col] - processed_df[imp_col]
    return processed_df

def generate_future_labels(last_period, steps, p_type):
    """Auto-detect Monthly / Yearly period format and generate dynamic future labels"""
    last_str = str(last_period).strip()

    # 1. နှစ် (Yearly) သက်သက် 4 လုံးပါရှိလျှင် (ဥပမာ - 2025, 2026)
    if last_str.isdigit() and len(last_str) == 4:
        start_y = int(last_str)
        return [str(start_y + i + 1) for i in range(steps)]

    # 2. လနှင့်နှစ် ပုံစံများကို Date အဖြစ် အလိုအလျောက် ဖြსတ်စစ်ခြင်း (ဥပမာ - Jan-26, 2026-01 စသည်)
    parsed_date = pd.to_datetime(last_str, format='%b-%y', errors='coerce')
    if pd.isna(parsed_date):
        parsed_date = pd.to_datetime(last_str, errors='coerce')
        
    if not pd.isna(parsed_date):
        if p_type == 'Monthly':
            # လအလိုက် ရှေ့သို့ တိုး၍ ရက်စွဲထုတ်ပေးခြင်း
            dates = pd.date_range(start=parsed_date + pd.DateOffset(months=1), periods=steps, freq='MS')
            return [d.strftime('%b-%y') for d in dates]
        else:
            # နှစ်အလိုက် ရှေ့သို့ တိုး၍ ထုတ်ပေးခြင်း
            years = range(parsed_date.year + 1, parsed_date.year + steps + 1)
            return [str(y) for y in years]

    # 3. အကယ်၍ Date လုံးဝ မဖတ်နိုင်မှသာ ပုံစံသစ်ဖြင့် ပြသရန် (M_Pred အစား ပိုသေသပ်သော ပုံစံ)
    prefix = "Month" if p_type == 'Monthly' else "Year"
    return [f"{prefix}+{i+1}" for i in range(steps)]

def apply_advanced_layout(fig, title_text, x_title, y_title):
    fig.update_layout(
        title=title_text,
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="LightGray",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        font=dict(family="sans-serif", size=12, color="#1e293b"),
        hovermode="x unified"
    )
    return fig

# ==========================================================
# PROFESSIONAL SIDEBAR DESIGN
# ==========================================================

st.sidebar.markdown("""
<div style="
background:linear-gradient(135deg,#2563eb,#06b6d4);
padding:18px;
border-radius:15px;
text-align:center;
margin-bottom:15px;
">
<h3 style='color:white;margin:0;'>
📊 AI Engineering Project of Group-3
</h3>
<p style='color:white;font-size:17px;margin-top:5px;'>
Advanced Data Analysis Dashboard
</p>
</div>
""", unsafe_allow_html=True)

import streamlit as st

st.markdown("""
<style>
.header-marquee {
    position: fixed;
    top: 12px;
    left: 255px; /* Sidebar နှင့် မထပ်စေရန် ညှိပါ */
    width: 75%;
    z-index: 999999; /* Header အပေါ်ဆုံးမှာ ပေါ်စေရန် */
    overflow: hidden;
    white-space: nowrap;
    pointer-events: none; /* Deploy ခလုတ် နှိပ်ရတာ မထိခိုက်စေရန် */
}

.header-marquee span {
    display: inline-block;
    padding-left: 100%;
    font-weight: bold;
    color: #1E3A8A;
    animation: marquee 20s linear infinite;
}

@keyframes marquee {
    0%   { transform: translate(0, 0); }
    100% { transform: translate(-100%, 0); }
}
</style>

<div class="header-marquee">
    <span style='color: #1E3A8A; font-weight: bold; font-size: 14px;font-family: Myanmar;'>
            🚀 AI Engineering Level-1 - သင်ကြားပို့ချပေးပါသော ဆရာမ ဒေါက်တာဇူးဇူး၊ ဆရာ သက်နိုင်ထွန်း နှင့် ITCSTC သင်တန်းကျောင်းမှ တာဝန်ရှိသူများအား ကျေးဇူး အထူးတင်ပါသည်။ <br> 
            Project Group-3 အဖွဲ့ဝင်များဖြစ်သော- ဒေါက်တာနွယ်နွယ်ဝင်း၊ ဦးဝေလင်းဦး၊ ဦးပြည့်ဖြိုးဟိန်း၊ ဦးထက်ပြည့်စုံဝင်း၊ ဒေါ်အေးချမ်းမိုး၊ ဒေါ်ဟေမာန်မိုးနှင့် ဒေါ်သီရိအောင် တို့ဖြစ်ပါသည်။</span>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<style>
            
/* Sidebar Background & Border */
section[data-testid="stSidebar"]{
    background-color:snowwhite;
    border-right:1px solid #334155;
}
            
/* Sidebar Text */
section[data-testid="stSidebar"] *{
    color:DarkBlue !important;
}
            
/* Expander Box */
div[data-testid="stExpander"]{
    border-radius:12px;
    border:1px solid #334155;
    background-color:#B7F188; /* greenleaf */
}
            
/* 1. Expander Header Hover အရောင်ကို အပြာရောင်ပြောင်းရန် */
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    background-color: white; !important; /* Soft Blue Hover */
    border-radius: 10px !important;
}

/* 2. Multiselect / Selectbox Item Hover အရောင်များကို အပြာရောင်ပြောင်းရန် */
div[data-baseweb="select"] ul li:hover,
div[data-baseweb="menu"] div:hover {
    background-color: #5CBDFF; !important; /* Professional Blue */
    color: #5CBDFF !important;
}

/* 3. Checkbox သို့မဟုတ် Radio items များ hover ဖြစ်ချိန် */
section[data-testid="stSidebar"] .stCheckbox:hover, 
section[data-testid="stSidebar"] .stRadio:hover {
    background-color: #B3C5D7 !important; /* Professional Blue */
    border-radius: 8px;
    padding: 2px 5px;
}
            
/* Selectbox / Multiselect */
div[data-baseweb="select"] > div{
    border-radius:10px !important;
    background-color: #B7F188 !important; /*greenleaf */
}
/* Metric */
div[data-testid="metric-container"]{
    background:#1e293b;
    border:1px solid #334155;
    padding:10px;
    border-radius:10px;
}
/* Buttons */
.stButton button{
    border-radius:10px;
}
            
/* File Uploader Box Text Styling */
[data-testid="stFileUploader"] section {
    background-color: #B7F188 !important;
}
[data-testid="stFileUploader"] div, [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] small {
    color: #1e293b !important; /* dataset upload Browser bg color */
}
[data-testid="stFileUploader"] button {
    color: #1e293b !important;
    background-color: #f1f5f9 !important;
    border: 3px solid #cbd5e1 !important;
}
            
/* Main Content Styling */
.main {
    background-color: #f8fafc;
}
.metric-card {
    background-color: #ffffff;
    padding: 0px;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    border: 20px solid #e2e8f0;
    border-left: 5px solid #2563eb;
    margin-bottom: 30px;
}
h1, h2, h3 {
    color: #1e293b;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)



# ==========================================================
# STEP 1 : DATA UPLOAD
# ==========================================================
st.sidebar.markdown("## ① Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df_raw_orig, df_clean = load_raw_and_clean_data(uploaded_file)
        st.sidebar.success("✅ Dataset Loaded Successfully")
    except Exception as e:
        st.sidebar.error(f"❌ Error : {e}")
        st.stop()
else:
    st.sidebar.info("Using Sample Dataset")
    dates = pd.date_range(
        start="2021-01-01",
        periods=36,
        freq="MS"
    )
    np.random.seed(42)
    df_clean = pd.DataFrame({
        "period": [d.strftime("%b-%y") for d in dates],
        "year": [d.year for d in dates],
        "product": np.random.choice(["Rice", "Bean", "Corn", "Rubber"], size=36),
        "export": np.linspace(500, 900, 36) + np.random.normal(0, 30, 36),
        "import": np.linspace(450, 800, 36) + np.random.normal(0, 40, 36),
        "local_consumption": np.linspace(200, 400, 36) + np.random.normal(0, 10, 36)
    })
    df_raw_orig = df_clean.copy()

if df_clean.empty:
    st.error("❌ တင်သွင်းလာသော Dataset ထဲတွင် အချက်အလက်များ လုံးဝမပါဝင်ပါ။")
    st.stop()

# ==========================================================
# STEP 2 : PERIOD SETTINGS
# ==========================================================
st.sidebar.markdown("## ② Time Settings")
default_x_idx = (
    df_clean.columns.tolist().index("period")
    if "period" in df_clean.columns else 0
)
x_col = st.sidebar.selectbox(
    "Period Column",
    df_clean.columns.tolist(),
    index=default_x_idx
)
period_type = st.sidebar.radio(
    "Period Type",
    ["Monthly", "Yearly"],
    horizontal=True
)

# ==========================================================
# STEP 3 : INDICATORS
# ==========================================================
st.sidebar.markdown("## ③ Indicators")
numeric_cols = df_clean.select_dtypes(
    include=["number"]
).columns.tolist()

default_cols = []
for col in numeric_cols:
    if col.lower() in ["export", "import"]:
        default_cols.append(col)
if len(default_cols) == 0 and numeric_cols:
    default_cols = [numeric_cols[0]]

selected_numerics = st.sidebar.multiselect(
    "Choose Numeric Variables",
    options=numeric_cols,
    default=default_cols
)

if len(selected_numerics) == 0:
    st.warning("Please select at least one indicator.")
    st.stop()

# ==========================================================
# STEP 4 : FILTERS
# ==========================================================
st.sidebar.markdown("## ④ Filters")
primary_num = selected_numerics[0]
min_thresh = float(df_clean[primary_num].min())
max_thresh = float(df_clean[primary_num].max())

st.sidebar.metric(
    "Current Indicator",
    primary_num
)

if min_thresh < max_thresh:
    threshold_val = st.sidebar.slider(
        f"{primary_num} Threshold",
        min_value=min_thresh,
        max_value=max_thresh,
        value=min_thresh
    )
else:
    threshold_val = min_thresh

# Category Filter
categorical_cols = df_clean.select_dtypes(
    include=["object", "category"]
).columns.tolist()

for col in numeric_cols:
    if df_clean[col].nunique() <= 15 and col not in categorical_cols and col not in selected_numerics:
        categorical_cols.append(col)

with st.sidebar.expander("🗂 Category Filters", expanded=False):
    active_cat_cols = st.multiselect(
        "Enable Filters",
        categorical_cols
    )

filtered_df = process_trade_data(df_clean)
filtered_df = filtered_df[filtered_df[primary_num] >= threshold_val]

for cat_col in active_cat_cols:
    values = sorted(
        filtered_df[cat_col].dropna().unique().tolist(),
        key=str
    )
    selected_vals = st.sidebar.multiselect(
        f"{cat_col}",
        values,
        default=values
    )
    filtered_df = filtered_df[filtered_df[cat_col].isin(selected_vals)]

# ==========================================================
# STEP 5 : ANALYSIS MODE
# ==========================================================
st.sidebar.markdown("## ⑤ Analysis")
analysis_mode = st.sidebar.selectbox(
    "Calculation Mode",
    [
        "Raw Values",
        "Mean Aggregation",
        "Percentage Change (%)"
    ]
)
group_by_col = st.sidebar.selectbox(
    "Group By",
    ["None"] + categorical_cols
)

if not filtered_df.empty:
    if analysis_mode == "Mean Aggregation":
        if group_by_col != 'None' and group_by_col != x_col:
            filtered_df = filtered_df.groupby([x_col, group_by_col], as_index=False)[selected_numerics].mean()
        else:
            filtered_df = filtered_df.groupby(x_col, as_index=False)[selected_numerics].mean()
    elif analysis_mode == "Percentage Change (%)":
        for col in selected_numerics:
            filtered_df[col] = filtered_df[col].pct_change().fillna(0) * 100
    else:
        if group_by_col != 'None' and group_by_col != x_col:
            filtered_df = filtered_df.groupby([x_col, group_by_col], as_index=False)[selected_numerics].sum()

    # Chronological Date / Month Sorting
    if period_type == 'Monthly':
        parsed_dates = pd.to_datetime(filtered_df[x_col], format='%b-%y', errors='coerce').fillna(
            pd.to_datetime(filtered_df[x_col], errors='coerce')
        )
        if not parsed_dates.isna().all():
            filtered_df['__sort_dt'] = parsed_dates
            filtered_df = filtered_df.sort_values(by='__sort_dt').drop(columns=['__sort_dt'])
        else:
            month_order = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 
                           'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
            if filtered_df[x_col].astype(str).str.lower().isin(month_order.keys()).any():
                filtered_df['__month_sort'] = filtered_df[x_col].astype(str).str.lower().map(month_order)
                filtered_df = filtered_df.sort_values(by='__month_sort').drop(columns=['__month_sort'])
    else:
        filtered_df = filtered_df.sort_values(by=x_col)

# ==========================================================
# STEP 6 : VISUALIZATION (Excel-like Combo Chart Customizer)
# ==========================================================
st.sidebar.markdown("## ⑥ Visualization & Combo Styles")
with st.sidebar.expander("📊 Excel-Style Combo Settings", expanded=True):
    st.markdown('<p style="font-size:13px; color:#e0e7ff; font-weight:600; font-style:italic">မိမိရွေးချယ်ထားသော Indicator တစ်ခုချင်းစီအတွက် Chart ပုံစံ (Line/Bar) ကို သီးသန့် သတ်မှတ်နိုင်ပါသည်။</p>', unsafe_allow_html=True)
    
    chart_type_mapping = {}
    for num_col in selected_numerics:
        chart_type_mapping[num_col] = st.selectbox(
            f"Type for: {num_col}",
            ["Line", "Bar", "Area"],
            key=f"chart_type_{num_col}"
        )
        
    st.markdown("---")
    show_labels = st.checkbox("Show Data Labels", value=True)
    show_ci = st.checkbox("95% CI Bands", value=True)
    custom_title = st.text_input("Chart Title")
    custom_xlabel = st.text_input("X Axis Label")
    custom_ylabel = st.text_input("Y Axis Label")

# ==========================================================
# STEP 7 : FORECASTING
# ==========================================================
st.sidebar.markdown("## ⑦ Forecasting")
with st.sidebar.expander("🔮 Forecast Settings", expanded=True):
    model_arima = st.checkbox("ARIMA")
    model_sarima = st.checkbox("SARIMA", value=True)
    model_prophet = st.checkbox("Prophet", value=False)
    model_xgboost = st.checkbox("XGBoost", value=False)
    model_linear = st.checkbox("Linear", value=True)
        
    forecast_steps = st.number_input("Forecast Horizon", min_value=1, max_value=24, value=12 if period_type == "Monthly" else 3)
    
    st.markdown("---")
    st.markdown("🎨 **Model Colors:**")
    color_arima = st.color_picker("ARIMA Color", value="#2563eb")
    color_sarima = st.color_picker("SARIMA Color", value="#636efa")
    color_prophet = st.color_picker("Prophet Color", value="#00cc96")
    color_xgboost = st.color_picker("XGBoost Color", value="#ab63fa")
    color_linear = st.color_picker("Linear Color", value="#ff7f0e")
# ==========================================================
# FOOTER
# ==========================================================
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align:center;'>
<b>Trade Analytics Studio</b>
<br>
<small>
Version 2.0
<br>
Powered by Streamlit & Python
</small>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Layout Structure using st.tabs
# ---------------------------------------------------------
tab_dash, tab_forecast, tab_insights, tab_data = st.tabs([
    "📊 Executive Dashboard",
    "🔮 Predictive Analytics",
    "🧠 Intelligence Center",
    "📂 Data Workspace"
])

# --- TAB 1: DASHBOARD & KPIS ---
with tab_dash:
    # st.header(f"1. 📊 Dashboard, KPIs & Trade Overview ({analysis_mode})") အစား -
    st.markdown(f"""
        <h2 style='font-size: 26px; font-weight: 700; color: darkblue; margin-bottom: 0px;'>
            1. 📊 Dashboard, KPIs & Trade Overview ({analysis_mode})
        </h2>
    """, unsafe_allow_html=True)
    
    if filtered_df.empty:
        st.warning("⚠️ ရွေးချယ်ထားသော Filter သို့မဟုတ် Threshold အတွင်း ဒေတာမရှိပါ။")
    else:
        latest_rec = filtered_df.iloc[-1]
        prev_rec = filtered_df.iloc[-2] if len(filtered_df) > 1 else latest_rec
        
        # 🌟 Executive KPI Cards များအတွက် Hover နှင့် Color Styling ထည့်သွင်းခြင်း
        st.markdown("""
            <style>
            .kpi-card {
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border: 1px solid rgba(255, 255, 255, 0.1);
                padding: 7px;
                border-radius: 5px;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                color: #ffffff;
                margin-bottom: 3px;
                transition: all 0.3s ease-in-out; /* 🌟 ပြောင်းလဲမှုများကို ချောမွေ့စေရန် */
            }
            
            /* 🌟 မောက်စ်တင်လိုက်စဉ် (Hover) ဖြစ်ပေါ်မည့် ပုံစံ */
            .kpi-card:hover {
                transform: translateY(-5px); /* ကဒ်လေး အနည်းငယ် ပုံဖော်ထောင်တက်လာမည် */
                border-color: rgba(52, 211, 153, 0.5); /* ဘောင်အရောင် တောက်ပလာမည် (အစိမ်းရောင်ဖျော့ဖျော့) */
                box-shadow: 0 12px 24px -6px rgba(0, 0, 0, 0.4), 0 0 20px rgba(52, 211, 153, 0.2); /* အရိပ်နှင့် မှုန်ဝါးသော အလင်းရောင် (Glow) ထွက်လာမည် */
            }

            .kpi-title {
                font-size: 16px;
                font-weight: 600;
                color: #94a3b8;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 2px;
            }
            .kpi-value {
                font-size: 14px;
                font-weight: 700;
                color: #f8fafc;
                margin-bottom: 2px;
            }
            .kpi-delta-pos {
                font-size: 14px;
                font-weight: 600;
                color: #34d399;
            }
            .kpi-delta-neg {
                font-size: 14px;
                font-weight: 600;
                color: #f87171;
            }
            </style>
        """, unsafe_allow_html=True)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        val_unit = "%" if analysis_mode == "Percentage Change (%)" else ""
        
        # 1st Metric
        val1 = latest_rec[selected_numerics[0]]
        prev1 = prev_rec[selected_numerics[0]]
        diff1 = ((val1 - prev1) / prev1 * 100) if prev1 != 0 else 0
        
        # 🌟 +/- နှင့် မျှားသင်္ကေတ နှစ်မျိုးစလုံး တွဲသုံးခြင်း (+↑ / -↓)
        sign_arrow1 = "+↑" if diff1 >= 0 else "-↓"
        delta_class1 = "kpi-delta-pos" if diff1 >= 0 else "kpi-delta-neg"
        
        with col_m1:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Latest {selected_numerics[0]}</div>
                    <div class="kpi-value">{val1:,.2f}{val_unit}</div>
                    <div class="{delta_class1}">{sign_arrow1} {abs(diff1):.2f}% vs previous</div>
                </div>
            """, unsafe_allow_html=True)
            
        # 2nd Metric
        if len(selected_numerics) >= 2:
            val2 = latest_rec[selected_numerics[1]]
            prev2 = prev_rec[selected_numerics[1]]
            diff2 = ((val2 - prev2) / prev2 * 100) if prev2 != 0 else 0
            
            sign_arrow2 = "+↑" if diff2 >= 0 else "-↓"
            delta_class2 = "kpi-delta-pos" if diff2 >= 0 else "kpi-delta-neg"
            
            with col_m2:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Latest {selected_numerics[1]}</div>
                        <div class="kpi-value">{val2:,.2f}{val_unit}</div>
                        <div class="{delta_class2}">{sign_arrow2} {abs(diff2):.2f}% vs previous</div>
                    </div>
                """, unsafe_allow_html=True)
        
        # 3rd Metric (Total Trade)
        if 'total_trade' in filtered_df.columns and analysis_mode == "Raw Values":
            tot_val = latest_rec['total_trade']
            prev_tot = prev_rec['total_trade']
            tot_diff = ((tot_val - prev_tot) / prev_tot * 100) if prev_tot != 0 else 0
            
            sign_arrow_tot = "+↑" if tot_diff >= 0 else "-↓"
            delta_class_tot = "kpi-delta-pos" if tot_diff >= 0 else "kpi-delta-neg"
            
            with col_m3:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Total Trade</div>
                        <div class="kpi-value">${tot_val:,.2f}M</div>
                        <div class="{delta_class_tot}">{sign_arrow_tot} {abs(tot_diff):.2f}% vs previous</div>
                    </div>
                """, unsafe_allow_html=True)
                
        # 4th Metric (Trade Balance)
        if 'trade_balance' in filtered_df.columns and analysis_mode == "Raw Values":
            bal_val = latest_rec['trade_balance']
            bal_status = "Surplus 🟢" if bal_val >= 0 else "Deficit 🔴"
            bal_color = "#34d399" if bal_val >= 0 else "#f87171"
            
            with col_m4:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Trade Balance</div>
                        <div class="kpi-value">${bal_val:,.2f}M</div>
                        <div style="font-size: 13px; font-weight: 600; color: {bal_color};">{bal_status}</div>
                    </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # Excel-Style Combo Chart Rendering for Tab 1
        # ---------------------------------------------------------
        fig_combo = go.Figure()
        x_vals = filtered_df[x_col].astype(str).tolist()
        palette = px.colors.qualitative.Plotly

        for idx, col_name in enumerate(selected_numerics):
            y_vals = filtered_df[col_name].astype(float).tolist()
            base_color = palette[idx % len(palette)]
            
            # Sidebar တွင် သတ်မှတ်ထားသော Chart Type (Line, Bar သို့မဟုတ် Area) ကို ယူခြင်း
            chosen_type = chart_type_mapping.get(col_name, "Line")
            
            if chosen_type == "Bar":
                fig_combo.add_trace(go.Bar(
                    x=x_vals, y=y_vals,
                    text=[f"{v:,.1f}" for v in y_vals] if show_labels else None,
                    textposition="auto",
                    name=f"{col_name} [Bar]",
                    marker=dict(color=base_color)
                ))
            elif chosen_type == "Area":
                fig_combo.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines+markers+text' if show_labels else 'lines+markers',
                    text=[f"{v:,.1f}" for v in y_vals] if show_labels else None,
                    textposition="top center",
                    name=f"{col_name} [Area]",
                    fill='tozeroy',
                    line=dict(width=2, color=base_color)
                ))
            else:  # Line (Default)
                fig_combo.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines+markers+text' if show_labels else 'lines+markers',
                    text=[f"{v:,.1f}" for v in y_vals] if show_labels else None,
                    textposition="top center",
                    name=f"{col_name} [Line]",
                    line=dict(width=3, color=base_color)
                ))

        # Layout ကို အဆင့်မြှင့်တင်ခြင်း
        fig_combo = apply_advanced_layout(
            fig_combo, 
            custom_title if custom_title else f"Trend & Combo Analysis [{analysis_mode}] (X: {x_col})",
            custom_xlabel if custom_xlabel else x_col,
            custom_ylabel if custom_ylabel else "Values"
        )

        # Chart ကို Streamlit တွင် ပြသခြင်း
        st.plotly_chart(fig_combo, use_container_width=True)

def apply_advanced_layout(fig, title_text, x_title, y_title):
    fig.update_layout(
        title=title_text,
        xaxis_title=x_title,
        yaxis_title=y_title,
        # 🌟 ဤနေရာတွင် X-axis ၏ အညွှန်းစာသားများကို လိုချင်သည့်ဒီဂရီ (ဥပမာ - 30 သို့မဟုတ် 45) စောင်းပေးခြင်း
        xaxis=dict(
            tickangle=120,  # 0, 30, 45, 90 စသဖြင့် လိုသလို ပေးနိုင်သည်
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="LightGray",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=80, b=40),
        font=dict(family="sans-serif", size=12, color="#1e293b"),
        hovermode="x unified"
    )
    return fig


# --- TAB 2: AI FORECASTING & MODELS ---
with tab_forecast:
    st.header(f"2. 🤖 {period_type} Analysis & Forecast (With Prophet & XGBoost)")

    if filtered_df.empty:
        st.warning("⚠️ ခန့်မှန်းချက်တွက်ချက်ရန် ဒေတာမရှိပါ။")
    else:
        fig = go.Figure()
        x_labels_hist = filtered_df[x_col].astype(str).tolist()
        palette = px.colors.qualitative.Plotly
        
        # 🌟 ဤနေရာတွင် အနာဂတ်ကာလ အညွှန်းတန်ဖိုးများကို ဦးစွာ ဖန်တီးသတ်မှတ်ပေးရပါမည်
        last_period_val = x_labels_hist[-1] if len(x_labels_hist) > 0 else "0"
        future_x_labels = generate_future_labels(last_period_val, forecast_steps, period_type)

        evaluation_metrics = []

        for idx, col_name in enumerate(selected_numerics):
            y_values = filtered_df[col_name].astype(float).tolist()
            base_color = palette[idx % len(palette)]
            
            # Excel-style Combo Chart Type (Line, Bar သို့မဟုတ် Area)
            chosen_type = chart_type_mapping.get(col_name, "Line")
            
            if chosen_type == "Bar":
                fig.add_trace(go.Bar(
                    x=x_labels_hist, y=y_values,
                    text=[f"{v:,.1f}" for v in y_values] if show_labels else None,
                    textposition="auto",
                    name=f"Historical ({col_name}) [Bar]",
                    marker=dict(color=base_color)
                ))
            elif chosen_type == "Area":
                fig.add_trace(go.Scatter(
                    x=x_labels_hist, y=y_values,
                    mode='lines+markers+text' if show_labels else 'lines+markers',
                    text=[f"{v:,.1f}" for v in y_values] if show_labels else None,
                    textposition="top center",
                    name=f"Historical ({col_name}) [Area]",
                    fill='tozeroy',
                    line=dict(width=2, color=base_color)
                ))
            else:  # Default to Line
                fig.add_trace(go.Scatter(
                    x=x_labels_hist, y=y_values,
                    mode='lines+markers+text' if show_labels else 'lines+markers',
                    text=[f"{v:,.1f}" for v in y_values] if show_labels else None,
                    textposition="top center",
                    name=f"Historical ({col_name}) [Line]",
                    line=dict(width=3, color=base_color)
                ))
            
            # 1. ARIMA
            if model_arima and len(y_values) >= 5:
                try:
                    ts_data = pd.Series(y_values)
                    arima_res = ARIMA(ts_data, order=(1, 1, 1)).fit()
                    fitted_vals = arima_res.fittedvalues
                    evaluation_metrics.append({
                        "Model": f"ARIMA (1,1,1) - {col_name}",
                        "MAE": mean_absolute_error(y_values, fitted_vals),
                        "RMSE": np.sqrt(mean_squared_error(y_values, fitted_vals))
                    })
                    forecast_obj = arima_res.get_forecast(steps=forecast_steps)
                    forecast_df = forecast_obj.summary_frame(alpha=0.05)
                    pred_y = forecast_df['mean'].tolist()

                    fig.add_trace(go.Scatter(
                        x=future_x_labels, y=pred_y,
                        mode='lines+markers+text' if show_labels else 'lines+markers',
                        text=[f"{v:,.1f}" for v in pred_y] if show_labels else None,
                        textposition="top center",
                        name=f"ARIMA Forecast ({col_name})",
                        line=dict(dash='dash', width=3, color=color_arima)
                    ))
                except Exception as e:
                    st.error(f"ARIMA Error [{col_name}]: {e}")

            # 2. SARIMA
            if model_sarima and len(y_values) >= 6:
                try:
                    ts_data = pd.Series(y_values)
                    s_order = (1, 1, 1, 12) if (period_type == 'Monthly' and len(y_values) >= 12) else (1, 1, 1, 2)
                    sarima_model = SARIMAX(ts_data, order=(1, 1, 1), seasonal_order=s_order)
                    sarima_res = sarima_model.fit(disp=False)
                    fitted_vals = sarima_res.fittedvalues
                    evaluation_metrics.append({
                        "Model": f"SARIMA - {col_name}",
                        "MAE": mean_absolute_error(y_values, fitted_vals),
                        "RMSE": np.sqrt(mean_squared_error(y_values, fitted_vals))
                    })
                    forecast_obj = sarima_res.get_forecast(steps=forecast_steps)
                    forecast_df = forecast_obj.summary_frame(alpha=0.05)
                    pred_y_sarima = forecast_df['mean'].tolist()

                    fig.add_trace(go.Scatter(
                        x=future_x_labels, y=pred_y_sarima,
                        mode='lines+markers+text' if show_labels else 'lines+markers',
                        text=[f"{v:,.1f}" for v in pred_y_sarima] if show_labels else None,
                        textposition="top center",
                        name=f"SARIMA Forecast ({col_name})",
                        line=dict(dash='dot', width=3, color=color_sarima)
                    ))
                except Exception as e:
                    st.error(f"SARIMA Error [{col_name}]: {e}")

            # 3. PROPHET
            if model_prophet and len(y_values) >= 3:
                try:
                    df_prophet = pd.DataFrame({
                        'ds': pd.date_range(start='2020-01-01', periods=len(y_values), freq='MS' if period_type == 'Monthly' else 'YS'),
                        'y': y_values
                    })
                    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
                    m.fit(df_prophet)
                    future = m.make_future_dataframe(periods=forecast_steps, freq='MS' if period_type == 'Monthly' else 'YS')
                    forecast = m.predict(future)
                    
                    pred_y_prophet = forecast['yhat'].tail(forecast_steps).tolist()
                    
                    fitted_prophet = forecast['yhat'].head(len(y_values)).tolist()
                    evaluation_metrics.append({
                        "Model": f"Prophet - {col_name}",
                        "MAE": mean_absolute_error(y_values, fitted_prophet),
                        "RMSE": np.sqrt(mean_squared_error(y_values, fitted_prophet))
                    })

                    fig.add_trace(go.Scatter(
                        x=future_x_labels, y=pred_y_prophet,
                        mode='lines+markers+text' if show_labels else 'lines+markers',
                        text=[f"{v:,.1f}" for v in pred_y_prophet] if show_labels else None,
                        textposition="top center",
                        name=f"Prophet Forecast ({col_name})",
                        line=dict(dash='longdash', width=3, color=color_prophet)
                    ))
                except Exception as e:
                    st.error(f"Prophet Error [{col_name}]: {e}")

            # 4. XGBOOST
            if model_xgboost and len(y_values) >= 4:
                try:
                    X_train = np.arange(len(y_values)).reshape(-1, 1)
                    y_train = np.array(y_values)
                    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                    xgb_model.fit(X_train, y_train)
                    
                    fitted_xgb = xgb_model.predict(X_train)
                    evaluation_metrics.append({
                        "Model": f"XGBoost - {col_name}",
                        "MAE": mean_absolute_error(y_train, fitted_xgb),
                        "RMSE": np.sqrt(mean_squared_error(y_train, fitted_xgb))
                    })

                    X_future = np.arange(len(y_values), len(y_values) + forecast_steps).reshape(-1, 1)
                    pred_y_xgb = xgb_model.predict(X_future).tolist()

                    fig.add_trace(go.Scatter(
                        x=future_x_labels, y=pred_y_xgb,
                        mode='lines+markers+text' if show_labels else 'lines+markers',
                        text=[f"{v:,.1f}" for v in pred_y_xgb] if show_labels else None,
                        textposition="bottom center",
                        name=f"XGBoost Forecast ({col_name})",
                        line=dict(dash='solid', width=3, color=color_xgboost)
                    ))
                except Exception as e:
                    st.error(f"XGBoost Error [{col_name}]: {e}")

            # 5. LINEAR REGRESSION
            if model_linear and len(y_values) >= 2:
                try:
                    X_train = np.arange(len(y_values)).reshape(-1, 1)
                    y_train = np.array(y_values)
                    reg = LinearRegression().fit(X_train, y_train)
                    pred_train = reg.predict(X_train)
                    
                    evaluation_metrics.append({
                        "Model": f"Linear Regression - {col_name}",
                        "MAE": mean_absolute_error(y_train, pred_train),
                        "RMSE": np.sqrt(mean_squared_error(y_train, pred_train))
                    })

                    X_future = np.arange(len(y_values), len(y_values) + forecast_steps).reshape(-1, 1)
                    pred_y_linear = reg.predict(X_future)

                    fig.add_trace(go.Scatter(
                        x=future_x_labels, y=pred_y_linear,
                        mode='lines+markers+text' if show_labels else 'lines+markers',
                        text=[f"{v:,.1f}" for v in pred_y_linear] if show_labels else None,
                        textposition="bottom center",
                        name=f"Linear Forecast ({col_name})",
                        line=dict(dash='dashdot', width=3, color=color_linear)
                    ))
                except Exception as e:
                    st.error(f"Linear Error [{col_name}]: {e}")

        fig = apply_advanced_layout(
            fig,
            custom_title if custom_title else f"Multi-Variable {period_type} Analysis & Forecast (With Prophet & XGBoost)",
            custom_xlabel if custom_xlabel else f"Period ({period_type})",
            custom_ylabel if custom_ylabel else "Values"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------------
        # Enhanced Model Evaluation Metrics Display Section
        # ---------------------------------------------------------
        if evaluation_metrics:
            st.markdown("---")
            st.subheader("📌 AI Model Performance Evaluation (MAE & RMSE Comparison)")
            st.markdown("အောက်ပါဇယားသည် မော်ဒယ် တစ်ခုချင်းစီ၏ ခန့်မှန်းချက် အမှားအယွင်း (Error) များကို တိုင်းတာထားခြင်း ဖြစ်ပါသည်။ **RMSE နှင့် MAE တန်ဖိုး နိမ့်လေ၊ မော်ဒယ် ပိုမိုတိကျလေ** ဖြစ်ပါသည်။")
            
            # DataFrame အဖြစ်ပြောင်းလဲပြီး RMSE ဖြင့် အသေးဆုံးမှ အကြီးဆုံးသို့ အစဉ်လိုက် စီခြင်း (Sorting)
            eval_df = pd.DataFrame(evaluation_metrics).sort_values(by="RMSE").reset_index(drop=True)
            
            def highlight_min(s):
                is_min = s == s.min()
                return ['background-color: #dbeafe; font-weight: bold;' if v else '' for v in is_min]

            styled_eval_df = eval_df.style.format({
                "MAE": "{:,.4f}", 
                "RMSE": "{:,.4f}"
            }).apply(highlight_min, subset=['MAE', 'RMSE'])

            st.table(styled_eval_df)
            
            best_model_name = eval_df.iloc[0]['Model']
            best_rmse = eval_df.iloc[0]['RMSE']
            st.success(f"🏆 **Best Performing Model:** ယခု ဒေတာအစုအဝေးအတွက် အတိကျဆုံး ခန့်မှန်းပေးနိုင်သည့် မော်ဒယ်မှာ **{best_model_name}** (RMSE: {best_rmse:,.4f}) ဖြစ်ပါသည်။")
# --- TAB 3: INTELLIGENT INSIGHTS ---
with tab_insights:

    st.markdown(
        '<div class="summary-header">🧠 Executive Intelligence Dashboard</div>',
        unsafe_allow_html=True
    )

    if not filtered_df.empty and len(selected_numerics) > 0:

        primary_col = selected_numerics[0]

        avg_val = filtered_df[primary_col].mean()
        max_val = filtered_df[primary_col].max()
        min_val = filtered_df[primary_col].min()
        total_val = filtered_df[primary_col].sum()
        median_val = filtered_df[primary_col].median()
        std_val = filtered_df[primary_col].std()

        latest_val = filtered_df[primary_col].iloc[-1]

        # --------------------
        # KPI ROW
        # --------------------

        k1, k2, k3, k4, k5, k6 = st.columns(6)

        with k1:
            st.markdown(f"""
            <div class="insight-kpi">
                <div class="insight-kpi-icon">📈</div>
                <div class="insight-kpi-title">Average</div>
                <div class="insight-kpi-value">{avg_val:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="insight-kpi">
                <div class="insight-kpi-icon">🚀</div>
                <div class="insight-kpi-title">Maximum</div>
                <div class="insight-kpi-value">{max_val:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="insight-kpi">
                <div class="insight-kpi-icon">📉</div>
                <div class="insight-kpi-title">Minimum</div>
                <div class="insight-kpi-value">{min_val:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            st.markdown(f"""
            <div class="insight-kpi">
                <div class="insight-kpi-icon">💰</div>
                <div class="insight-kpi-title">Total</div>
                <div class="insight-kpi-value">{total_val:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with k5:
            st.markdown(f"""
            <div class="insight-kpi">
                <div class="insight-kpi-icon">🎯</div>
                <div class="insight-kpi-title">Median</div>
                <div class="insight-kpi-value">{median_val:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with k6:
            st.markdown(f"""
            <div class="insight-kpi">
                <div class="insight-kpi-icon">📊</div>
                <div class="insight-kpi-title">Std Dev</div>
                <div class="insight-kpi-value">{std_val:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # --------------------
        # Executive Insights
        # --------------------

        trend_status = (
            "Strong Growth 📈"
            if latest_val > avg_val
            else "Below Average 📉"
        )

        data_quality = (
            filtered_df[primary_col]
            .isna()
            .sum()
        )

        left, right = st.columns(2)

        with left:

            st.markdown(f"""
            <div class='insight-card'>
            <h4>📋 Executive Summary</h4>

            <b>Primary Metric:</b> {primary_col}<br>

            <b>Analysis Mode:</b> {analysis_mode}<br>

            <b>Records Analyzed:</b> {len(filtered_df):,}<br>

            <b>Current Trend:</b> {trend_status}<br>

            <b>Latest Value:</b> {latest_val:,.2f}
            </div>
            """,
            unsafe_allow_html=True)

        with right:

            st.markdown(f"""
            <div class='insight-card'>
            <h4>🔍 Data Quality Assessment</h4>

            <b>Missing Values:</b> {data_quality}<br>

            <b>Maximum:</b> {max_val:,.2f}<br>

            <b>Minimum:</b> {min_val:,.2f}<br>

            <b>Volatility:</b> {std_val:,.2f}
            </div>
            """,
            unsafe_allow_html=True)

        # --------------------
        # Distribution Metrics
        # --------------------

        st.subheader("📊 Metric Distribution Analysis")

        fig_dist = px.histogram(
            filtered_df,
            x=primary_col,
            marginal="box",
            color_discrete_sequence=["#2563eb"]
        )

        fig_dist.update_layout(
            height=450,
            template="plotly_white"
        )

        st.plotly_chart(
            fig_dist,
            use_container_width=True
        )

        # --------------------
        # Correlation Matrix
        # --------------------

        if len(selected_numerics) >= 2:

            st.subheader("🔗 Correlation Intelligence Matrix")

            corr = filtered_df[selected_numerics].corr()

            fig_corr = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="RdBu_r"
            )

            fig_corr.update_layout(
                height=600,
                template="plotly_white"
            )

            st.plotly_chart(
                fig_corr,
                use_container_width=True
            )

# --- TAB 4: DATA PREVIEW & EXPORT ---
with tab_data:
    st.header("4. 📄 Data Preview (Raw vs Clean) & Styled Excel Export")
    
    # Toggle between Raw Data and Clean Processed Data (Matches Jupyter ToggleButton)
    show_raw_mode = st.toggle("🔄 Show Raw Original Dataset (Unfiltered)", value=False)
    
    if show_raw_mode:
        st.subheader("📋 Raw Dataset Preview")
        st.dataframe(df_raw_orig, use_container_width=True)
        export_df = df_raw_orig.copy()
    else:
        st.subheader("✨ Clean & Filtered Chart Summary Data")
        st.dataframe(filtered_df, use_container_width=True)
        export_df = filtered_df.copy()

    col1, col2 = st.columns(2)
    
    # Styled Excel Export using xlsxwriter (Matches Jupyter export_to_excel styling)
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        export_df.to_excel(writer, sheet_name='Chart_Summary_Data', index=False)
        workbook = writer.book
        worksheet = writer.sheets['Chart_Summary_Data']
        
        header_format = workbook.add_format({
            'bold': True, 
            'fg_color': '#DCE6F1', 
            'border': 1
        })
        for col_num, value in enumerate(export_df.columns.values):
            worksheet.write(0, col_num, str(value), header_format)
            worksheet.set_column(col_num, col_num, 18)
            
    excel_buffer.seek(0)

    with col1:
        st.download_button(
            label="📥 Export Chart Data (Styled Excel)",
            data=excel_buffer,
            file_name="chart_summary_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    with col2:
        csv_data = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Chart Data (CSV)",
            data=csv_data,
            file_name="chart_summary_data.csv",
            mime="text/csv"
        )
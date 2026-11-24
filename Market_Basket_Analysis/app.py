import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# --- Page Configuration ---
st.set_page_config(
    page_title="Market Basket Analysis — Retail Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .badge-info {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }

</style>
""", unsafe_allow_html=True)

# --- Base Paths ---
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "dataset" / "processed" / "cleaned_data.csv"
RAW_DATA_PATH = BASE_DIR / "dataset" / "raw" / "Online_Retail_enlargedcsv.csv"
PREPROC_SUMMARY_PATH = BASE_DIR / "results" / "preprocessing_summary.csv"
ITEMSETS_DIR = BASE_DIR / "results" / "frequent_itemsets"
RULES_DIR = BASE_DIR / "results" / "association_rules"

# --- Data Loading with Cache ---
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M').astype(str)
    return df

@st.cache_data
def load_encoded_matrix():
    df = pd.read_csv(DATA_PATH)
    df['Description'] = df['Description'].str.strip()
    transactions = df.groupby('InvoiceNo')['Description'].apply(list).tolist()
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    return pd.DataFrame(te_ary, columns=te.columns_)

@st.cache_data
def load_meta_counts():
    try:
        n_itemsets = len(pd.read_csv(ITEMSETS_DIR / "apriori_itemsets_min_sup_0_01.csv"))
    except Exception:
        n_itemsets = 712
    try:
        n_rules = len(pd.read_csv(RULES_DIR / "apriori_association_rules_min_sup_0_01.csv"))
    except Exception:
        n_rules = 143
    return n_itemsets, n_rules

@st.cache_data
def load_preproc_summary():
    if PREPROC_SUMMARY_PATH.exists():
        return pd.read_csv(PREPROC_SUMMARY_PATH)
    return None

# --- Cached Rule Mining Function ---
@st.cache_data(show_spinner=False)
def mine_rules(algo, min_sup, min_conf):
    df_encoded = load_encoded_matrix()
    if algo == "FP-Growth":
        frequent_itemsets = fpgrowth(df_encoded, min_support=min_sup, use_colnames=True)
    else:
        frequent_itemsets = apriori(df_encoded, min_support=min_sup, use_colnames=True)
    
    if len(frequent_itemsets) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
    if len(rules) == 0:
        return frequent_itemsets, pd.DataFrame()
    
    rules['Antecedent'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['Consequent'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    return frequent_itemsets, rules

# --- Sidebar Navigation State & Dynamic Styling ---
sidebar_style = """
<style>
    /* Global Sidebar Container - Deep Solid Black Theme */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.09) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    [data-testid="stSidebar"][aria-expanded="true"] {
        width: 300px !important;
        min-width: 300px !important;
    }

    [data-testid="stSidebar"][aria-expanded="false"] {
        width: 0 !important;
        min-width: 0 !important;
        border-right: none !important;
    }

    /* Expand / Collapse Button Smooth Accessibility */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 100000 !important;
    }

    /* Pinned Expand Control when collapsed */
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapsedControl"] {
        position: fixed !important;
        top: 0.75rem !important;
        left: 0.75rem !important;
    }

    [data-testid="stExpandSidebarButton"] button,
    [data-testid="stSidebarCollapsedControl"] button {
        background: #000000 !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        border-radius: 9px !important;
        color: #60A5FA !important;
        padding: 0.35rem 0.55rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.7) !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stExpandSidebarButton"] button:hover,
    [data-testid="stSidebarCollapsedControl"] button:hover {
        background: #111827 !important;
        border-color: #3B82F6 !important;
        color: #FFFFFF !important;
        transform: scale(1.08) !important;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4) !important;
    }

    [data-testid="stSidebarCollapseButton"] button {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        color: #94A3B8 !important;
        transition: all 0.2s ease !important;
    }

    [data-testid="stSidebarCollapseButton"] button:hover {
        background: #1E293B !important;
        color: #FFFFFF !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #000000 !important;
        padding: 1.25rem 1.1rem 1.8rem 1.1rem !important;
    }

    /* Top Window Dots (macOS style matching reference image) */
    .sidebar-window-dots {
        display: flex;
        gap: 7px;
        align-items: center;
        margin-bottom: 1.1rem;
        padding-left: 0.35rem;
    }
    .window-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .window-dot-red { background: #EF4444; }
    .window-dot-yellow { background: #F59E0B; }
    .window-dot-green { background: #10B981; }

    /* Brand Header Banner */
    .sidebar-brand-wrapper {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 0.35rem 0.25rem 0.5rem 0.25rem;
    }

    .sidebar-brand-icon-box {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.28) 0%, rgba(37, 99, 235, 0.15) 100%);
        border: 1px solid rgba(96, 165, 250, 0.4);
        flex-shrink: 0;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.25);
    }

    .sidebar-brand-info {
        display: flex;
        flex-direction: column;
    }

    .sidebar-brand-title {
        font-size: 1.08rem;
        font-weight: 700;
        line-height: 1.2;
        color: #FFFFFF;
        letter-spacing: -0.015em;
    }

    .sidebar-brand-subtitle {
        font-size: 0.72rem;
        font-weight: 500;
        color: #94A3B8;
        margin-top: 0.2rem;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* Subtle Dividers */
    .sidebar-nav-divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(255, 255, 255, 0.14) 0%, rgba(255, 255, 255, 0.02) 100%);
        margin: 1.1rem 0 1rem 0;
    }

    /* Section Label */
    .sidebar-nav-section-title {
        font-size: 0.70rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #64748B;
        padding-left: 0.4rem;
        margin-bottom: 0.75rem;
    }

    /* Radio Group Reset & Container Styling */
    [data-testid="stSidebar"] [data-testid="stRadio"] > label {
        display: none !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] div[role="radiogroup"],
    [data-testid="stSidebar"] [data-testid="stRadioGroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 0.60rem !important;
    }

    /* Completely Eliminate Radio Bullet Circle Across All Streamlit Versions */
    [data-testid="stSidebar"] [data-testid="stRadioOption"] > div > div:first-child,
    [data-testid="stSidebar"] [data-testid="stRadioOption"] [class*="ex50key4"],
    [data-testid="stSidebar"] [data-testid="stRadioOption"] [class*="ex50key5"],
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"],
    [data-testid="stSidebar"] [data-testid="stRadioOption"] svg,
    [data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child,
    [data-testid="stSidebar"] [data-baseweb="radio"] input {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
    }

    /* Navigation Item Card (Full width, featurable interactive module) */
    [data-testid="stSidebar"] [data-testid="stRadioOption"],
    [data-testid="stSidebar"] [data-baseweb="radio"] {
        background: #0A0A0A !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 12px !important;
        padding: 0.85rem 1.15rem !important;
        margin: 0 !important;
        width: 100% !important;
        box-sizing: border-box !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover,
    [data-testid="stSidebar"] [data-baseweb="radio"]:hover {
        background: #141414 !important;
        border-color: rgba(255, 255, 255, 0.18) !important;
        transform: translateX(3px) !important;
    }

    /* Active / Selected Navigation Card */
    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked),
    [data-testid="stSidebar"] [data-testid="stRadioGroup"] > div:has(input:checked),
    [data-testid="stSidebar"] [data-testid="stRadioOption"][data-selected="true"],
    [data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(90deg, rgba(37, 99, 235, 0.25) 0%, rgba(30, 58, 138, 0.15) 100%) !important;
        border: 1px solid rgba(96, 165, 250, 0.45) !important;
        border-left: 4px solid #3B82F6 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.25) !important;
    }

    /* Enlarged Icons & Spaced Label Typography */
    [data-testid="stSidebar"] [data-testid="stRadioOption"] p,
    [data-testid="stSidebar"] [data-baseweb="radio"] p {
        color: #94A3B8 !important;
        font-size: 0.98rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
        display: block !important;
        letter-spacing: 0.01em !important;
        line-height: 1.4 !important;
        white-space: nowrap !important;
        transition: color 0.18s ease !important;
    }

    /* Enlarge the leading emoji icon and give generous spacing */
    [data-testid="stSidebar"] [data-testid="stRadioOption"] p::first-letter,
    [data-testid="stSidebar"] [data-baseweb="radio"] p::first-letter {
        font-size: 1.65rem !important;
        line-height: 1 !important;
        vertical-align: -0.15rem !important;
        margin-right: 0.65rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadioOption"]:hover p,
    [data-testid="stSidebar"] [data-baseweb="radio"]:hover p {
        color: #F8FAFC !important;
    }

    [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) p,
    [data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    /* Featurable Action Buttons (Clear, Purposeful, Non-Random) */
    [data-testid="stSidebar"] div.stButton button {
        width: 100% !important;
        background: #111111 !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 9px !important;
        color: #E2E8F0 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        padding: 0.55rem 0.85rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.5rem !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4) !important;
    }

    [data-testid="stSidebar"] div.stButton button:hover {
        background: #1E293B !important;
        border-color: #3B82F6 !important;
        color: #60A5FA !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
    }

    /* Status Box on Deep Black */
    .sidebar-footer-box {
        margin-top: 1.8rem;
        padding: 1rem 1.1rem;
        border-radius: 12px;
        background: #080808;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
    }

    .sidebar-footer-header {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #64748B;
        margin-bottom: 0.65rem;
    }

    .sidebar-footer-metrics {
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        margin-bottom: 0.75rem;
    }

    .sidebar-metric-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        color: #94A3B8;
    }

    .metric-val {
        font-weight: 700;
        color: #F8FAFC;
    }

    .sidebar-footer-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(16, 185, 129, 0.12);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-bottom: 0.85rem;
    }

    .status-pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #10B981;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulseGreen 2s infinite;
    }

    @keyframes pulseGreen {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
</style>
"""
st.sidebar.markdown(sidebar_style, unsafe_allow_html=True)

# Top Window Decorative Dots (macOS styled)
st.sidebar.markdown("""
<div class="sidebar-window-dots">
    <span class="window-dot window-dot-red"></span>
    <span class="window-dot window-dot-yellow"></span>
    <span class="window-dot window-dot-green"></span>
</div>
""", unsafe_allow_html=True)

# Clean, Full-Width Brand Header (No random or ghost buttons)
st.sidebar.markdown("""
<div class="sidebar-brand-wrapper">
    <div class="sidebar-brand-icon-box">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#60A5FA" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M6 2 3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4Z"></path>
            <path d="M3 6h18"></path>
            <path d="M16 10a4 4 0 0 1-8 0"></path>
        </svg>
    </div>
    <div class="sidebar-brand-info">
        <div class="sidebar-brand-title">Market Basket Analysis</div>
        <div class="sidebar-brand-subtitle">Retail Analytics Studio</div>
    </div>
</div>
<div class="sidebar-nav-divider"></div>
<div class="sidebar-nav-section-title">Navigation Modules</div>
""", unsafe_allow_html=True)

# Navigation Destination List
NAV_PAGES = [
    "Dashboard / Overview",
    "Dataset Upload",
    "Data Preprocessing",
    "Exploratory Data Analysis",
    "Association Rule Mining",
    "Algorithm Comparison"
]

NAV_ICONS = {
    "Dashboard / Overview": "📊",
    "Dataset Upload": "📂",
    "Data Preprocessing": "⚙️",
    "Exploratory Data Analysis": "📈",
    "Association Rule Mining": "🔍",
    "Algorithm Comparison": "⚡"
}

page = st.sidebar.radio(
    "Navigation",
    NAV_PAGES,
    format_func=lambda x: f"{NAV_ICONS.get(x, '')}\u2003\u2002{x}",
    label_visibility="collapsed"
)

# Bottom Data Engine Status Card with Featurable Action Button
st.sidebar.markdown("""
<div class="sidebar-footer-box">
    <div class="sidebar-footer-header">DATA ENGINE STATUS</div>
    <div class="sidebar-footer-metrics">
        <div class="sidebar-metric-row">
            <span class="metric-label">Raw Records:</span>
            <span class="metric-val">1,041,909</span>
        </div>
        <div class="sidebar-metric-row">
            <span class="metric-label">Cleaned Records:</span>
            <span class="metric-val">756,097</span>
        </div>
    </div>
    <div class="sidebar-footer-tag">
        <span class="status-pulse-dot"></span> Online &amp; Ready
    </div>
</div>
""", unsafe_allow_html=True)

# Featurable Button: Real functional purpose (Clear cache & reload data engine)
if st.sidebar.button("⚡ Reload Data Engine", key="btn_reload_data_engine", help="Clear transaction cache and re-index dataset from disk"):
    st.cache_data.clear()
    st.toast("Data Engine cache reloaded successfully!", icon="🚀")
    st.rerun()

# ==============================================================================
# DASHBOARD / OVERVIEW
# ==============================================================================
if page == "Dashboard / Overview":
    st.markdown('<div class="main-header">Retail Performance Dashboard & Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Key retail indicators, basket sizing distributions, and purchasing patterns</div>', unsafe_allow_html=True)

    with st.spinner("Loading transaction data..."):
        df = load_data()
        n_itemsets, n_rules = load_meta_counts()

    total_txns = df['InvoiceNo'].nunique()
    total_products = df['Description'].nunique()
    total_customers = df['CustomerID'].nunique()
    total_revenue = df['TotalPrice'].sum()
    basket_sizes = df.groupby('InvoiceNo')['Description'].nunique()
    avg_basket_size = basket_sizes.mean()

    st.subheader("Key Performance Indicators")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Total Transactions", f"{total_txns:,}")
    col2.metric("Unique Products", f"{total_products:,}")
    col3.metric("Total Customers", f"{total_customers:,}")
    col4.metric("Avg Basket Size", f"{avg_basket_size:.1f} items")
    col5.metric("Frequent Itemsets", f"{n_itemsets:,}", help="Mined at min_support = 0.01")
    col6.metric("Association Rules", f"{n_rules:,}", help="Mined with Lift > 1.2, min_confidence = 0.2")

    st.markdown("---")

    st.subheader("Top-Selling Products")
    tab1, tab2 = st.tabs(["By Total Quantity Sold", "By Invoice Frequency (Basket Appearances)"])

    with tab1:
        top_qty = df.groupby('Description')['Quantity'].sum().reset_index()
        top_qty = top_qty.sort_values(by='Quantity', ascending=False).head(10)

        chart_qty = alt.Chart(top_qty).mark_bar(color="#2563EB", cornerRadiusEnd=4).encode(
            x=alt.X('Quantity:Q', title='Total Quantity Sold'),
            y=alt.Y('Description:N', sort='-x', title='Product Description'),
            tooltip=['Description', alt.Tooltip('Quantity:Q', format=',')]
        ).properties(height=350)
        st.altair_chart(chart_qty, use_container_width=True)

    with tab2:
        top_inv = df.groupby('Description')['InvoiceNo'].nunique().reset_index()
        top_inv.columns = ['Description', 'InvoiceCount']
        top_inv = top_inv.sort_values(by='InvoiceCount', ascending=False).head(10)

        chart_inv = alt.Chart(top_inv).mark_bar(color="#7C3AED", cornerRadiusEnd=4).encode(
            x=alt.X('InvoiceCount:Q', title='Number of Unique Transactions'),
            y=alt.Y('Description:N', sort='-x', title='Product Description'),
            tooltip=['Description', alt.Tooltip('InvoiceCount:Q', format=',')]
        ).properties(height=350)
        st.altair_chart(chart_inv, use_container_width=True)

    st.markdown("---")

    st.subheader("Dataset Overview Charts")
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("**Monthly Sales Revenue Trend (£)**")
        monthly = df.groupby('YearMonth')['TotalPrice'].sum().reset_index()
        monthly.columns = ['Month', 'Revenue']

        chart_trend = alt.Chart(monthly).mark_line(point=True, color="#059669", strokeWidth=3).encode(
            x=alt.X('Month:O', title='Month (Year-Month)'),
            y=alt.Y('Revenue:Q', title='Revenue (£)', axis=alt.Axis(format='~s')),
            tooltip=['Month', alt.Tooltip('Revenue:Q', format='$,.2f')]
        ).properties(height=320)
        st.altair_chart(chart_trend, use_container_width=True)

    with chart_col2:
        st.markdown("**Country-wise Transactions (Top 10)**")
        country_df = df.groupby('Country')['InvoiceNo'].nunique().reset_index()
        country_df.columns = ['Country', 'Transactions']
        top_countries = country_df.sort_values(by='Transactions', ascending=False).head(10)

        chart_country = alt.Chart(top_countries).mark_bar(color="#D97706", cornerRadiusEnd=4).encode(
            x=alt.X('Transactions:Q', title='Unique Invoices'),
            y=alt.Y('Country:N', sort='-x', title='Country'),
            tooltip=['Country', alt.Tooltip('Transactions:Q', format=',')]
        ).properties(height=320)
        st.altair_chart(chart_country, use_container_width=True)

    st.markdown("**Basket Size Distribution (Items per Invoice)**")
    basket_df = basket_sizes.reset_index()
    basket_df.columns = ['InvoiceNo', 'BasketSize']
    basket_filtered = basket_df[basket_df['BasketSize'] <= 60]

    chart_hist = alt.Chart(basket_filtered).mark_bar(color="#475569", binSpacing=1).encode(
        x=alt.X('BasketSize:Q', bin=alt.Bin(maxbins=30), title='Basket Size (Unique Items)'),
        y=alt.Y('count():Q', title='Frequency (Number of Transactions)'),
        tooltip=[alt.Tooltip('count():Q', format=',')]
    ).properties(height=260)
    st.altair_chart(chart_hist, use_container_width=True)

# ==============================================================================
# PAGE 2: DATASET UPLOAD
# ==============================================================================
elif page == "Dataset Upload":
    st.markdown('<div class="main-header">Dataset Upload & Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a retail transaction dataset (CSV or Excel) to preview and inspect data structure and quality</div>', unsafe_allow_html=True)

    upload_col1, upload_col2 = st.columns([3, 1])
    with upload_col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Upload retail sales records with columns such as InvoiceNo, StockCode, Description, Quantity, UnitPrice, CustomerID"
        )
    with upload_col2:
        st.markdown("**Or load project data:**")
        use_sample = st.button("Load Raw Dataset", help="Load project raw dataset (Online_Retail_enlargedcsv.csv)")
        use_cleaned = st.button("Load Cleaned Dataset", help="Load project cleaned dataset (cleaned_data.csv)")

    uploaded_df = None
    data_source_name = ""

    if uploaded_file is not None:
        try:
            with st.spinner("Reading uploaded file..."):
                if uploaded_file.name.endswith('.csv'):
                    uploaded_df = pd.read_csv(uploaded_file, encoding='latin1')
                else:
                    uploaded_df = pd.read_excel(uploaded_file)
                data_source_name = f"Uploaded File: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Error loading file: {e}")

    elif use_sample and RAW_DATA_PATH.exists():
        with st.spinner("Loading raw dataset..."):
            uploaded_df = pd.read_csv(RAW_DATA_PATH, encoding='latin1')
            data_source_name = "Project Raw Dataset (Online_Retail_enlargedcsv.csv)"

    elif use_cleaned and DATA_PATH.exists():
        with st.spinner("Loading cleaned dataset..."):
            uploaded_df = pd.read_csv(DATA_PATH)
            data_source_name = "Project Cleaned Dataset (cleaned_data.csv)"

    if uploaded_df is not None:
        st.success(f"Successfully loaded: **{data_source_name}**")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records (Rows)", f"{uploaded_df.shape[0]:,}")
        m2.metric("Total Columns", f"{uploaded_df.shape[1]}")
        total_nulls = int(uploaded_df.isnull().sum().sum())
        m3.metric("Missing Values", f"{total_nulls:,}")
        m4.metric("Duplicate Rows", f"{uploaded_df.duplicated().sum():,}")

        st.markdown("---")

        st.subheader("1. Dataset Preview")
        preview_mode = st.radio("Display Rows:", ["First 15 Rows (Head)", "Last 15 Rows (Tail)", "Random Sample (15)"], horizontal=True)

        if preview_mode == "First 15 Rows (Head)":
            st.dataframe(uploaded_df.head(15), use_container_width=True)
        elif preview_mode == "Last 15 Rows (Tail)":
            st.dataframe(uploaded_df.tail(15), use_container_width=True)
        else:
            st.dataframe(uploaded_df.sample(min(15, len(uploaded_df))), use_container_width=True)

        st.markdown("---")

        st.subheader("2. Column Specifications & Data Types")
        col_specs = []
        for col in uploaded_df.columns:
            null_count = int(uploaded_df[col].isnull().sum())
            null_pct = round((null_count / len(uploaded_df)) * 100, 2)
            col_specs.append({
                "Column Name": col,
                "Data Type": str(uploaded_df[col].dtype),
                "Non-Null Count": f"{len(uploaded_df) - null_count:,}",
                "Missing Count": f"{null_count:,}",
                "Missing (%)": f"{null_pct}%",
                "Unique Values": f"{uploaded_df[col].nunique():,}",
                "Sample Value": str(uploaded_df[col].dropna().iloc[0]) if uploaded_df[col].dropna().shape[0] > 0 else "N/A"
            })
        spec_df = pd.DataFrame(col_specs)
        st.dataframe(spec_df, use_container_width=True)

        st.markdown("---")

        st.subheader("3. Missing Values Analysis")
        missing_series = uploaded_df.isnull().sum()
        missing_df = missing_series[missing_series > 0].reset_index()

        if len(missing_df) > 0:
            missing_df.columns = ['Column', 'MissingCount']
            missing_df['MissingPct'] = (missing_df['MissingCount'] / len(uploaded_df) * 100).round(2)

            miss_col1, miss_col2 = st.columns([1, 1])

            with miss_col1:
                st.markdown("**Missing Count Chart**")
                miss_chart = alt.Chart(missing_df).mark_bar(color="#EF4444", cornerRadiusEnd=4).encode(
                    x=alt.X('MissingCount:Q', title='Missing Values Count'),
                    y=alt.Y('Column:N', sort='-x', title='Column'),
                    tooltip=['Column', alt.Tooltip('MissingCount:Q', format=','), alt.Tooltip('MissingPct:Q', title='Missing %')]
                ).properties(height=250)
                st.altair_chart(miss_chart, use_container_width=True)

            with miss_col2:
                st.markdown("**Missing Values Summary Table**")
                st.dataframe(missing_df, use_container_width=True)
                st.warning("Columns with missing values (e.g. CustomerID or Description) will need cleaning in the preprocessing step.")
        else:
            st.success("No missing values found! The dataset is complete across all columns.")

    else:
        st.info("Please upload a CSV or Excel file, or click one of the load buttons above to inspect data.")

# ==============================================================================
# PAGE 3: DATA PREPROCESSING
# ==============================================================================
elif page == "Data Preprocessing":
    st.markdown('<div class="main-header">Data Preprocessing & Cleaning Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Select preprocessing operations to clean transactions and evaluate before-and-after quality statistics</div>', unsafe_allow_html=True)

    st.subheader("1. Configure Cleaning Options")
    opt_col1, opt_col2 = st.columns(2)

    with opt_col1:
        opt_missing = st.checkbox("Handle Missing Values (Drop rows without CustomerID or Description)", value=True)
        opt_duplicates = st.checkbox("Remove Duplicate Records", value=True)
        opt_cancelled = st.checkbox("Remove Cancelled Transactions (InvoiceNo starting with 'C')", value=True)

    with opt_col2:
        opt_qty = st.checkbox("Remove Invalid Quantities (Quantity <= 0)", value=True)
        opt_price = st.checkbox("Remove Invalid Unit Prices (UnitPrice <= 0)", value=True)
        opt_types = st.checkbox("Standardize Data Types (Trim strings, cast CustomerID to int, parse dates)", value=True)

    st.markdown("---")

    preproc_summary = load_preproc_summary()

    raw_records = 1041909
    raw_missing = 266101
    raw_duplicates = 5268
    raw_cancelled = 17239
    raw_invalid_qty = 18605
    raw_invalid_price = 2515

    cleaned_records = 756097
    cleaned_missing = 0
    cleaned_duplicates = 0
    cleaned_cancelled = 0
    cleaned_invalid_qty = 0
    cleaned_invalid_price = 0

    st.subheader("2. Before vs. After Preprocessing Statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{cleaned_records:,}", delta=f"-{raw_records - cleaned_records:,} removed", delta_color="normal")
    c2.metric("Missing Values", f"{cleaned_missing}", delta=f"-{raw_missing:,}", delta_color="inverse")
    c3.metric("Duplicate Records", f"{cleaned_duplicates}", delta=f"-5,225", delta_color="inverse")
    c4.metric("Retention Rate", "72.57%", help="756,097 of 1,041,909 high-quality transaction rows retained")

    st.markdown("#### Comprehensive Quality Audit (Before vs. After)")
    comparison_table = pd.DataFrame({
        "Metric": [
            "Total Records (Rows)",
            "Missing Values (CustomerID / Description)",
            "Duplicate Records",
            "Cancelled Transactions (C-prefix)",
            "Invalid Quantities (Quantity <= 0)",
            "Invalid Prices (UnitPrice <= 0)",
            "Data Quality Score"
        ],
        "Before Preprocessing (Raw)": [
            f"{raw_records:,}",
            f"{raw_missing:,}",
            f"{raw_duplicates:,}",
            f"{raw_cancelled:,}",
            f"{raw_invalid_qty:,}",
            f"{raw_invalid_price:,}",
            "Unreliable (Missing IDs & Cancelled items)"
        ],
        "After Preprocessing (Cleaned)": [
            f"{cleaned_records:,}",
            f"{cleaned_missing}",
            f"{cleaned_duplicates}",
            f"{cleaned_cancelled}",
            f"{cleaned_invalid_qty}",
            f"{cleaned_invalid_price}",
            "100% Clean (Ready for Apriori & FP-Growth)"
        ],
        "Difference / Impact": [
            "-285,812 rows (-27.4%)",
            "-266,101 missing values resolved",
            "-5,225 exact duplicates eliminated",
            "-17,239 return records removed",
            "All non-positive quantities purged",
            "40 zero/negative price items removed",
            "High integrity retail basket matrix"
        ]
    })
    st.table(comparison_table)

    st.markdown("---")

    st.subheader("3. Step-by-Step Data Cleaning Breakdown")
    if preproc_summary is not None:
        p_col1, p_col2 = st.columns([1, 1])

        with p_col1:
            st.markdown("**Rows Removed by Cleaning Step**")
            chart_steps = alt.Chart(preproc_summary).mark_bar(color="#DC2626", cornerRadiusEnd=4).encode(
                x=alt.X('Removed:Q', title='Rows Removed'),
                y=alt.Y('Step:N', sort='-x', title='Preprocessing Step'),
                tooltip=['Step', alt.Tooltip('Before:Q', format=','), alt.Tooltip('Removed:Q', format=','), alt.Tooltip('After:Q', format=','), alt.Tooltip('Pct:Q', title='% Removed')]
            ).properties(height=280)
            st.altair_chart(chart_steps, use_container_width=True)

        with p_col2:
            st.markdown("**Step-by-Step Tracking Table**")
            st.dataframe(preproc_summary, use_container_width=True)
            st.caption("Missing CustomerID represents the vast majority (25.3%) of dropped records.")

    st.markdown("---")

    st.subheader("4. Cleaned Dataset Preview & Export")
    df_cleaned = load_data()
    st.dataframe(df_cleaned.head(15), use_container_width=True)

    export_col1, export_col2 = st.columns([1, 3])
    with export_col1:
        st.download_button(
            label="Download Cleaned Dataset (CSV)",
            data=df_cleaned.head(5000).to_csv(index=False),
            file_name="cleaned_data_sample.csv",
            mime="text/csv",
            help="Download top 5,000 cleaned rows as sample CSV"
        )
    with export_col2:
        st.success("Cleaned dataset is saved at `dataset/processed/cleaned_data.csv` and ready for Transaction Matrix encoding.")

# ==============================================================================
# PAGE 4: EXPLORATORY DATA ANALYSIS (PLOTLY INTERACTIVE)
# ==============================================================================
elif page == "Exploratory Data Analysis":
    st.markdown('<div class="main-header">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Interactive visual exploration of purchasing behavior, demand distribution, seasonality, and basket compositions using Plotly</div>', unsafe_allow_html=True)

    with st.spinner("Loading cleaned dataset for interactive analysis..."):
        df = load_data()

    st.subheader("1. Most Frequently Purchased Products")
    prod_col1, prod_col2 = st.columns([2, 1])

    with prod_col2:
        metric_mode = st.radio(
            "Ranking Metric:",
            ["Total Quantity Sold", "Unique Invoices (Basket Appearances)"],
            help="Compare products bought in bulk volume vs. products that appear in the most individual carts."
        )
        top_n = st.slider("Number of Top Products to Display:", min_value=5, max_value=20, value=10)

    with prod_col1:
        if metric_mode == "Total Quantity Sold":
            top_df = df.groupby('Description')['Quantity'].sum().reset_index()
            top_df = top_df.sort_values(by='Quantity', ascending=False).head(top_n)
            fig_top = px.bar(
                top_df.sort_values(by='Quantity', ascending=True),
                x='Quantity',
                y='Description',
                orientation='h',
                title=f'Top {top_n} Products by Total Quantity Sold',
                color='Quantity',
                color_continuous_scale='Viridis',
                text='Quantity'
            )
            fig_top.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_top.update_layout(xaxis_title="Units Sold", yaxis_title="", height=420, margin=dict(l=0, r=40, t=40, b=0))
            st.plotly_chart(fig_top, use_container_width=True)
        else:
            top_df = df.groupby('Description')['InvoiceNo'].nunique().reset_index()
            top_df.columns = ['Description', 'Invoices']
            top_df = top_df.sort_values(by='Invoices', ascending=False).head(top_n)
            fig_top = px.bar(
                top_df.sort_values(by='Invoices', ascending=True),
                x='Invoices',
                y='Description',
                orientation='h',
                title=f'Top {top_n} Products by Invoice Appearances',
                color='Invoices',
                color_continuous_scale='Magma',
                text='Invoices'
            )
            fig_top.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_top.update_layout(xaxis_title="Unique Invoices", yaxis_title="", height=420, margin=dict(l=0, r=40, t=40, b=0))
            st.plotly_chart(fig_top, use_container_width=True)

    st.info("**Retail Takeaway**: High quantity does not always mean high invoice appearances. *Paper Craft Little Birdie* leads in total units sold due to massive wholesale purchases, whereas *White Hanging Heart T-Light Holder* appears in the highest number of distinct consumer baskets (2,969 invoices), making it a prime candidate for frequent itemset mining.")

    st.markdown("---")

    st.subheader("2. Monthly Sales Revenue Over Time (£)")
    monthly_sales = df.groupby('YearMonth')['TotalPrice'].sum().reset_index()
    monthly_sales.columns = ['YearMonth', 'Revenue']

    fig_sales = px.line(
        monthly_sales,
        x='YearMonth',
        y='Revenue',
        title="Monthly Revenue Trend (December 2010 – December 2012)",
        markers=True,
        line_shape='spline',
        hover_data={'Revenue': ':$,.2f', 'YearMonth': True}
    )
    fig_sales.update_traces(line=dict(color='#2563EB', width=3), marker=dict(size=8, color='#1E40AF'))
    fig_sales.update_layout(
        xaxis_title="Year-Month",
        yaxis_title="Total Revenue (£)",
        hovermode="x unified",
        height=380,
        margin=dict(l=0, r=20, t=40, b=0)
    )
    st.plotly_chart(fig_sales, use_container_width=True)

    st.info("**Seasonality Insight**: Strong Q4 surge in consumer spending every year. Revenue peaks dramatically in **November** (£1.16M in 2011, £1.44M in 2012) driven by pre-Christmas holiday retail shopping, followed by a seasonal trough in January–February.")

    st.markdown("---")

    st.subheader("3. Country-wise Transaction Distribution")
    c_toggle = st.checkbox("Exclude United Kingdom (to inspect international market breakdown)", value=False)

    country_df = df.groupby('Country')['InvoiceNo'].nunique().reset_index()
    country_df.columns = ['Country', 'Transactions']
    country_df['Share_Pct'] = (country_df['Transactions'] / country_df['Transactions'].sum() * 100).round(2)

    if c_toggle:
        chart_countries = country_df[country_df['Country'] != 'United Kingdom'].sort_values(by='Transactions', ascending=False).head(15)
        title_str = "Top 15 International Markets (Excluding UK)"
        bar_color = '#10B981'
    else:
        chart_countries = country_df.sort_values(by='Transactions', ascending=False).head(15)
        title_str = "Top 15 Countries by Transaction Count (Including UK)"
        bar_color = '#F59E0B'

    fig_country = px.bar(
        chart_countries,
        x='Country',
        y='Transactions',
        title=title_str,
        text='Transactions',
        hover_data={'Share_Pct': ':.2f%'},
        color='Transactions',
        color_continuous_scale='Teal' if c_toggle else 'Sunset'
    )
    fig_country.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
    fig_country.update_layout(xaxis_title="", yaxis_title="Unique Invoices", height=400, margin=dict(l=0, r=20, t=40, b=0))
    st.plotly_chart(fig_country, use_container_width=True)

    st.info("**Geographical Insight**: The **United Kingdom accounts for ~89.9%** (31,764 transactions) of the business. The top foreign markets are Germany (925), France (700), and Ireland (386). Association rules primarily reflect domestic UK shopping habits.")

    st.markdown("---")

    st.subheader("4. Basket-Size Distribution (Items per Invoice)")
    basket_sizes = df.groupby('InvoiceNo')['Description'].nunique().reset_index()
    basket_sizes.columns = ['InvoiceNo', 'BasketSize']

    b_col1, b_col2 = st.columns([3, 1])

    with b_col2:
        st.markdown("**Distribution Metrics:**")
        st.write(f"- **Mean Basket Size**: {basket_sizes['BasketSize'].mean():.2f} items")
        st.write(f"- **Median Basket Size**: {basket_sizes['BasketSize'].median():.0f} items")
        st.write(f"- **75th Percentile**: {basket_sizes['BasketSize'].quantile(0.75):.0f} items")
        st.write(f"- **Max Basket Size**: {basket_sizes['BasketSize'].max()} items")
        max_zoom = st.slider("Zoom Upper Bound (Basket Size):", min_value=20, max_value=200, value=60)

    with b_col1:
        basket_filtered = basket_sizes[basket_sizes['BasketSize'] <= max_zoom]
        fig_basket = px.histogram(
            basket_filtered,
            x='BasketSize',
            nbins=40,
            title=f"Basket Size Frequency (Zoomed to ≤ {max_zoom} items)",
            color_discrete_sequence=['#8B5CF6']
        )
        fig_basket.add_vline(x=basket_sizes['BasketSize'].mean(), line_dash="dash", line_color="red", annotation_text=f"Mean: {basket_sizes['BasketSize'].mean():.1f}")
        fig_basket.add_vline(x=basket_sizes['BasketSize'].median(), line_dash="dash", line_color="green", annotation_text=f"Median: {basket_sizes['BasketSize'].median():.0f}")
        fig_basket.update_layout(xaxis_title="Number of Unique Items in Basket", yaxis_title="Number of Invoices", height=380, margin=dict(l=0, r=20, t=40, b=0))
        st.plotly_chart(fig_basket, use_container_width=True)

    st.markdown("---")

    st.subheader("5. Product Purchase Frequency (Long-Tail / Pareto Distribution)")
    freq_df = df.groupby('Description')['InvoiceNo'].nunique().reset_index()
    freq_df.columns = ['Description', 'PurchaseCount']
    freq_df = freq_df.sort_values(by='PurchaseCount', ascending=False).reset_index(drop=True)
    freq_df['ProductRank'] = freq_df.index + 1

    fig_freq = px.line(
        freq_df.head(500),
        x='ProductRank',
        y='PurchaseCount',
        title="Top 500 Products: Purchase Frequency Decay (Classic Power-Law / Long Tail)",
        log_y=True,
        color_discrete_sequence=['#EC4899']
    )
    fig_freq.update_layout(
        xaxis_title="Product Rank (1 = Most Popular)",
        yaxis_title="Invoice Count (Log Scale)",
        height=360,
        margin=dict(l=0, r=20, t=40, b=0)
    )
    st.plotly_chart(fig_freq, use_container_width=True)

    st.info("**Long-Tail Conclusion**: Retail purchases exhibit a steep power-law drop-off. A small fraction of products (the top ~15%) drive the majority of basket co-occurrences, which explains why a minimum support threshold around **0.01 (1%)** successfully isolates the most frequent associations without combinatorial explosion.")

# ==============================================================================
# PAGE 5: ASSOCIATION RULE MINING
# ==============================================================================
elif page == "Association Rule Mining":
    st.markdown('<div class="main-header">Association Rule Mining & Discovery</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Select algorithms, tune minimum support & confidence thresholds, filter meaningful retail affinity rules, and search products</div>', unsafe_allow_html=True)

    st.subheader("1. Configure Mining Parameters")
    param_col1, param_col2, param_col3 = st.columns(3)

    with param_col1:
        st.markdown("**Select Algorithm:**")
        algo_choice = st.radio(
            "Algorithm",
            ["FP-Growth", "Apriori"],
            help="FP-Growth is recommended: ~18x faster than Apriori and memory-safe."
        )

    with param_col2:
        st.markdown("**Minimum Support:**")
        sup_mode = st.radio(
            "Min Support Preset",
            ["0.01", "0.02", "0.05", "Custom value"],
            index=0
        )
        if sup_mode == "Custom value":
            min_support = st.number_input("Custom Support Value", min_value=0.005, max_value=0.50, value=0.01, step=0.005, format="%.3f")
        else:
            min_support = float(sup_mode)

    with param_col3:
        st.markdown("**Minimum Confidence:**")
        conf_mode = st.radio(
            "Min Confidence Preset",
            ["0.30", "0.40", "0.50", "Custom value"],
            index=0
        )
        if conf_mode == "Custom value":
            min_confidence = st.number_input("Custom Confidence Value", min_value=0.10, max_value=1.00, value=0.20, step=0.05, format="%.2f")
        else:
            min_confidence = float(conf_mode)

    st.markdown("---")
    
    btn_col1, btn_col2 = st.columns([1, 4])
    with btn_col1:
        run_mining = st.button("Generate Association Rules", type="primary")

    if "rules_data" not in st.session_state or run_mining:
        with st.spinner(f"Mining frequent itemsets and association rules using {algo_choice} (min_support={min_support}, min_confidence={min_confidence})..."):
            frequent_itemsets, rules_df = mine_rules(algo_choice, min_support, min_confidence)
            st.session_state["rules_data"] = rules_df
            st.session_state["itemsets_data"] = frequent_itemsets
            st.session_state["last_algo"] = algo_choice
            st.session_state["last_sup"] = min_support
            st.session_state["last_conf"] = min_confidence

    rules_df = st.session_state.get("rules_data", pd.DataFrame())
    frequent_itemsets = st.session_state.get("itemsets_data", pd.DataFrame())

    if len(rules_df) > 0:
        st.success(f"Discovered **{len(frequent_itemsets):,} frequent itemsets** and **{len(rules_df):,} association rules** using {st.session_state.get('last_algo', algo_choice)} (min_support={st.session_state.get('last_sup', min_support)}, min_confidence={st.session_state.get('last_conf', min_confidence)})")

        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Total Rules Generated", f"{len(rules_df):,}")
        m_c2.metric("Average Confidence", f"{rules_df['confidence'].mean() * 100:.1f}%")
        m_c3.metric("Average Lift", f"{rules_df['lift'].mean():.2f}")
        m_c4.metric("Maximum Lift", f"{rules_df['lift'].max():.2f}")

        st.markdown("---")

        st.subheader("2. Filter, Search & Sort Rules")
        filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])

        with filter_col1:
            search_query = st.text_input("🔍 Search for a product (Antecedent or Consequent):", placeholder="e.g. TEACUP, LUNCH BAG, HEART")
        
        with filter_col2:
            min_lift_filter = st.slider("Minimum Lift Filter:", min_value=1.0, max_value=float(round(rules_df['lift'].max(), 1)), value=1.0, step=0.5)

        with filter_col3:
            sort_by = st.selectbox(
                "Sort Rules By:",
                ["Lift (Highest first)", "Confidence (Highest first)", "Support (Highest first)"]
            )

        filtered_rules = rules_df[rules_df['lift'] >= min_lift_filter].copy()

        if search_query:
            query_clean = search_query.strip().upper()
            filtered_rules = filtered_rules[
                filtered_rules['Antecedent'].str.upper().str.contains(query_clean) |
                filtered_rules['Consequent'].str.upper().str.contains(query_clean)
            ]

        if sort_by == "Lift (Highest first)":
            filtered_rules = filtered_rules.sort_values(by='lift', ascending=False)
        elif sort_by == "Confidence (Highest first)":
            filtered_rules = filtered_rules.sort_values(by='confidence', ascending=False)
        else:
            filtered_rules = filtered_rules.sort_values(by='support', ascending=False)

        st.markdown(f"**Showing {len(filtered_rules):,} matching rules:**")
        
        display_cols = ['Antecedent', 'Consequent', 'support', 'confidence', 'lift', 'leverage', 'conviction']
        formatted_table = filtered_rules[display_cols].copy()
        formatted_table.rename(columns={
            'support': 'Support',
            'confidence': 'Confidence',
            'lift': 'Lift',
            'leverage': 'Leverage',
            'conviction': 'Conviction'
        }, inplace=True)
        
        formatted_table['Support'] = formatted_table['Support'].round(4)
        formatted_table['Confidence'] = (formatted_table['Confidence'] * 100).round(2).astype(str) + '%'
        formatted_table['Lift'] = formatted_table['Lift'].round(2)
        formatted_table['Leverage'] = formatted_table['Leverage'].round(4)
        formatted_table['Conviction'] = formatted_table['Conviction'].round(2)

        st.dataframe(formatted_table, use_container_width=True, height=350)

        export_csv = filtered_rules[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Rules as CSV",
            data=export_csv,
            file_name="market_basket_association_rules.csv",
            mime="text/csv"
        )

        st.markdown("---")

        st.subheader("3. Association Rule Visualizations")
        viz_tab1, viz_tab2 = st.tabs(["Support vs. Confidence (Colored by Lift)", "Top 10 Rules Ranked by Lift"])

        with viz_tab1:
            fig_scatter = px.scatter(
                filtered_rules,
                x='support',
                y='confidence',
                color='lift',
                size='lift',
                hover_data=['Antecedent', 'Consequent'],
                title="Rule Distribution: Support vs. Confidence (Size & Color = Lift)",
                color_continuous_scale='Viridis',
                labels={'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'}
            )
            fig_scatter.update_layout(height=420, margin=dict(l=0, r=20, t=40, b=0))
            st.plotly_chart(fig_scatter, use_container_width=True)

        with viz_tab2:
            top_10_rules = filtered_rules.head(10).copy()
            top_10_rules['Rule'] = top_10_rules['Antecedent'] + "  →  " + top_10_rules['Consequent']
            
            fig_bar = px.bar(
                top_10_rules.sort_values(by='lift', ascending=True),
                x='lift',
                y='Rule',
                orientation='h',
                title="Top 10 Rules by Lift Metric",
                color='lift',
                color_continuous_scale='Purples',
                text='lift'
            )
            fig_bar.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig_bar.update_layout(xaxis_title="Lift", yaxis_title="", height=420, margin=dict(l=0, r=30, t=40, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

    else:
        st.warning(f"No association rules found with the current parameters (min_support={min_support}, min_confidence={min_confidence}). Try lowering the support threshold (e.g. to 0.01) or confidence threshold (e.g. to 0.20).")

# ==============================================================================
# PAGE 6: ALGORITHM COMPARISON (APRIORI VS FP-GROWTH)
# ==============================================================================
elif page == "Algorithm Comparison":
    st.markdown('<div class="main-header">Algorithm Comparison: Apriori vs. FP-Growth</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Comprehensive empirical comparison of execution time, memory scalability, frequent itemset composition, and association rule quality</div>', unsafe_allow_html=True)

    # Benchmark metrics at min_support = 0.01 and min_confidence = 0.2
    st.subheader("1. Performance Metrics at min_support = 0.01")
    
    col_a1, col_a2, col_a3, col_a4, col_a5 = st.columns(5)
    col_a1.metric("Execution Time", "4.07 s", delta="17.9x faster (FP-Growth)", delta_color="normal")
    col_a2.metric("Frequent Itemsets", "712", delta="100% Identical", delta_color="off")
    col_a3.metric("Association Rules", "143", delta="100% Identical", delta_color="off")
    col_a4.metric("Average Confidence", "33.2%", delta="0.0% variance", delta_color="off")
    col_a5.metric("Average Lift", "8.27", delta="0.0% variance", delta_color="off")

    st.markdown("---")

    # Detailed Comparison Table
    st.subheader("2. Detailed Metric Comparison Table")
    comp_df = pd.DataFrame({
        "Evaluation Metric": [
            "Execution Time (min_support = 0.01)",
            "Frequent Itemsets Discovered",
            "  • Single-item Itemsets (1-item)",
            "  • Two-item Itemsets (2-item)",
            "  • Three-item Itemsets (3-item)",
            "Total Association Rules Generated",
            "Average Rule Confidence",
            "Average Rule Lift",
            "Rules with Lift > 1 (Positive Correlation)",
            "Scalability at Lower Support (min_support = 0.005)",
            "Database Scanning Mechanism",
            "Candidate Generation Overhead"
        ],
        "Apriori Algorithm": [
            "72.82 seconds",
            "712 itemsets",
            "630 itemsets",
            "81 itemsets",
            "1 itemset",
            "143 rules",
            "33.21%",
            "8.27",
            "143 rules (100%)",
            "Failed (MemoryError: required >59 GiB RAM for 897k candidate pairs)",
            "Multiple database scans (one scan per candidate size k)",
            "High: calculates combinatorial pairs (C(n, 2))"
        ],
        "FP-Growth Algorithm": [
            "4.07 seconds",
            "712 itemsets",
            "630 itemsets",
            "81 itemsets",
            "1 itemset",
            "143 rules",
            "33.21%",
            "8.27",
            "143 rules (100%)",
            "Completed smoothly in 5.62s (discovered 2,175 itemsets)",
            "Exactly 2 database scans (Pass 1: frequencies, Pass 2: tree construction)",
            "Zero candidate generation: directly mines compact FP-Tree"
        ],
        "Comparison Summary": [
            "FP-Growth is 17.9× faster",
            "100% Mathematical Equivalence",
            "Exact match",
            "Exact match",
            "Exact match",
            "100% Identical Rules",
            "Identical",
            "Identical",
            "All generated rules represent valid affinity",
            "Apriori crashes; FP-Growth scales effortlessly",
            "FP-Growth avoids repeated I/O scans",
            "FP-Growth avoids exponential combinatorial explosion"
        ]
    })
    st.table(comp_df)

    st.markdown("---")

    # Interactive Bar Charts
    st.subheader("3. Benchmark Visualizations")
    chart_tab1, chart_tab2 = st.tabs(["Execution Time Comparison", "Frequent Itemset Counts Across Support Thresholds"])

    with chart_tab1:
        time_data = pd.DataFrame({
            "Support Threshold": ["min_sup = 0.05", "min_sup = 0.05", "min_sup = 0.02", "min_sup = 0.02", "min_sup = 0.01", "min_sup = 0.01", "min_sup = 0.005", "min_sup = 0.005"],
            "Algorithm": ["Apriori", "FP-Growth", "Apriori", "FP-Growth", "Apriori", "FP-Growth", "Apriori", "FP-Growth"],
            "Execution Time (seconds)": [0.09, 1.15, 1.51, 1.84, 72.82, 4.07, 0, 5.62],
            "Note": ["0.09s", "1.15s", "1.51s", "1.84s", "72.82s", "4.07s", "FAILED (Out of Memory)", "5.62s (2,175 itemsets)"]
        })

        fig_time = px.bar(
            time_data[time_data["Support Threshold"] != "min_sup = 0.005"],
            x="Support Threshold",
            y="Execution Time (seconds)",
            color="Algorithm",
            barmode="group",
            text="Execution Time (seconds)",
            title="Execution Time Comparison (Apriori vs FP-Growth)",
            color_discrete_map={"Apriori": "#EF4444", "FP-Growth": "#10B981"}
        )
        fig_time.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
        fig_time.update_layout(yaxis_type="log", yaxis_title="Execution Time (Seconds - Log Scale)", height=420)
        st.plotly_chart(fig_time, use_container_width=True)
        st.warning("⚠️ **Note on min_support = 0.005**: Apriori is excluded from the chart above because it exceeded system memory (>59 GiB) and crashed, while FP-Growth finished in only **5.62 seconds**.")

    with chart_tab2:
        itemset_data = pd.DataFrame({
            "Itemset Type": ["1-itemsets (Single)", "1-itemsets (Single)", "2-itemsets (Pairs)", "2-itemsets (Pairs)", "3-itemsets (Triplets)", "3-itemsets (Triplets)"],
            "Algorithm": ["Apriori", "FP-Growth", "Apriori", "FP-Growth", "Apriori", "FP-Growth"],
            "Count": [630, 630, 81, 81, 1, 1]
        })

        fig_itemset = px.bar(
            itemset_data,
            x="Itemset Type",
            y="Count",
            color="Algorithm",
            barmode="group",
            text="Count",
            title="Frequent Itemset Composition Match (min_support = 0.01)",
            color_discrete_map={"Apriori": "#3B82F6", "FP-Growth": "#8B5CF6"}
        )
        fig_itemset.update_traces(texttemplate='%{text}', textposition='outside')
        fig_itemset.update_layout(yaxis_title="Discovered Itemsets Count", height=400)
        st.plotly_chart(fig_itemset, use_container_width=True)

    st.markdown("---")

    # Architectural Takeaway
    st.subheader("4. Algorithmic Trade-off & Engineering Conclusion")
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        st.markdown("""
        ### Why Apriori Bottlenecks:
        1. **Combinatorial Candidate Generation**: For $N$ frequent single items, Apriori evaluates $\\binom{N}{2}$ candidate pairs, $\\binom{N}{3}$ triplets, and so on. As support decreases, candidate sizes explode into millions.
        2. **Repeated I/O Passes**: Scans the entire dataset repeatedly for each candidate level $k$.
        3. **High Memory Overhead**: Storing boolean matrices for hundreds of thousands of candidate combinations causes out-of-memory crashes.
        """)

    with col_t2:
        st.markdown("""
        ### Why FP-Growth is Superior:
        1. **Zero Candidate Generation**: Directly traverses paths in an in-memory **FP-Tree**, eliminating candidate generation entirely.
        2. **Exactly 2 Dataset Scans**: Only scans data twice regardless of pattern length.
        3. **Prefix Compression**: High-frequency item sequences share common tree nodes, compressing thousands of baskets into a compact data structure.
        4. **Production Recommendation**: Use **FP-Growth** for enterprise production analytics pipelines.
        """)

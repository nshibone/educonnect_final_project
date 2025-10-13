import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIG ---
st.set_page_config(page_title="🏨 Airbnb Hotel - Upgraded Dashboard", layout="wide")
st.title("🏨 Airbnb Hotel Analysis — Upgraded Dashboard")

# ---------------------------------------------------
# DATA CLEANING FUNCTIONS
# ---------------------------------------------------

def clean_price(x):
    """Robustly cleans price/currency strings."""
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return x
    s = str(x).replace("$", "").replace("USD", "").strip()
    # Handle common thousand separators (comma or period) and decimal points
    try:
        # Try US format (e.g., 1,234.56)
        s_us = s.replace(",", "")
        return float(s_us)
    except ValueError:
        try:
            # Try European format (e.g., 1.234,56 -> 1234.56)
            s_eu = s.replace(".", "").replace(",", ".")
            return float(s_eu)
        except ValueError:
            return np.nan

def clean_percent(x):
    """Cleans percentage strings."""
    if pd.isna(x): return np.nan
    s = str(x).replace("%", "").replace(",", ".").strip()
    try:
        return float(s)
    except:
        return np.nan

def safe_column(df, name, alt_names=[]):
    """Finds a column based on a list of potential names."""
    for n in [name] + alt_names:
        if n in df.columns:
            return n
    return None

# ---------------------------------------------------
# FILE UPLOAD & INITIAL LOAD
# ---------------------------------------------------

# ⚠️ CRITICAL FIX: Replaced hardcoded path with file uploader
uploaded_file = st.sidebar.file_uploader("📂 Upload Airbnb/Hotel Data CSV", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file in the sidebar to begin the analysis.")
    st.stop()

@st.cache_data
def load_data(file):
    """Loads and caches data from the uploaded file."""
    return pd.read_csv(file, low_memory=False)

df = load_data(uploaded_file)

# ---------------------------------------------------
# COLUMN MAPPING (SIDEBAR)
# ---------------------------------------------------
st.sidebar.header("Map Dataset Columns")

suggestions = {
    "City": safe_column(df, "city", ["City", "location", "neighbourhood"]),
    "Area": safe_column(df, "area", ["area", "neighbourhood", "neighborhood", "district", "zone"]),
    "Price": safe_column(df, "price", ["Price", "price_usd", "daily_price"]),
    "Income": safe_column(df, "sales", ["revenue", "income", "earnings"]),
    "Reviewers": safe_column(df, "total reviewers number", ["review_count", "number_of_reviews"]),
    "Host Response": safe_column(df, "host response rate", ["host_response_rate"]),
    "Host Acceptance": safe_column(df, "host acceptance rate", ["host_acceptance_rate"])
}

cols = ["None"] + [str(c) for c in df.columns]

# Sidebar select boxes
def select_with_suggestion(label, suggest):
    idx = cols.index(str(suggest)) if str(suggest) in cols else 0
    return st.sidebar.selectbox(label, cols, index=idx)

CITY_COL = select_with_suggestion("City column", suggestions["City"])
AREA_COL = select_with_suggestion("Area column", suggestions["Area"])
PRICE_COL = select_with_suggestion("Price column", suggestions["Price"])
INCOME_COL = select_with_suggestion("Income / Sales column", suggestions["Income"])
REVIEWERS_COL = select_with_suggestion("Reviewers column", suggestions["Reviewers"])
HOST_RESP_COL = select_with_suggestion("Host Response Rate column", suggestions["Host Response"])
HOST_ACCEPT_COL = select_with_suggestion("Host Acceptance Rate column", suggestions["Host Acceptance"])

# Convert “None” string to Python None
def none_or_value(val): return None if val == "None" else val
MAPPED_COLS = {
    "CITY_COL": none_or_value(CITY_COL),
    "AREA_COL": none_or_value(AREA_COL),
    "PRICE_COL": none_or_value(PRICE_COL),
    "INCOME_COL": none_or_value(INCOME_COL),
    "REVIEWERS_COL": none_or_value(REVIEWERS_COL),
    "HOST_RESP_COL": none_or_value(HOST_RESP_COL),
    "HOST_ACCEPT_COL": none_or_value(HOST_ACCEPT_COL),
}

st.sidebar.markdown("### 🔎 Current mappings")
st.sidebar.json(MAPPED_COLS)

# ---------------------------------------------------
# DATA CLEANING AND PRE-PROCESSING (CACHED)
# ---------------------------------------------------

@st.cache_data
def clean_and_process_data(df, cols_map):
    """
    Applies all cleaning and creates new features.
    This is cached and only re-runs if the column mapping changes.
    """
    d = df.copy()
    
    # Apply cleaning functions
    if cols_map["PRICE_COL"]: d[cols_map["PRICE_COL"]] = d[cols_map["PRICE_COL"]].apply(clean_price)
    if cols_map["INCOME_COL"]: d[cols_map["INCOME_COL"]] = d[cols_map["INCOME_COL"]].apply(clean_price)
    if cols_map["REVIEWERS_COL"]: d[cols_map["REVIEWERS_COL"]] = pd.to_numeric(d[cols_map["REVIEWERS_COL"]], errors="coerce")
    if cols_map["HOST_RESP_COL"]: d[cols_map["HOST_RESP_COL"]] = d[cols_map["HOST_RESP_COL"]].apply(clean_percent)
    if cols_map["HOST_ACCEPT_COL"]: d[cols_map["HOST_ACCEPT_COL"]] = d[cols_map["HOST_ACCEPT_COL"]].apply(clean_percent)
    
    return d

d = clean_and_process_data(df, MAPPED_COLS)
PRICE_COL, INCOME_COL, REVIEWERS_COL, HOST_RESP_COL, HOST_ACCEPT_COL = (
    MAPPED_COLS["PRICE_COL"], MAPPED_COLS["INCOME_COL"], MAPPED_COLS["REVIEWERS_COL"], 
    MAPPED_COLS["HOST_RESP_COL"], MAPPED_COLS["HOST_ACCEPT_COL"]
)
CITY_COL, AREA_COL = MAPPED_COLS["CITY_COL"], MAPPED_COLS["AREA_COL"]

# ---------------------------------------------------
# FILTERS (SIDEBAR)
# ---------------------------------------------------
st.sidebar.header("🔍 Filters")

# Area filter
if AREA_COL:
    areas = sorted(d[AREA_COL].dropna().unique().tolist())
    area_options = ["All"] + areas
    selected_areas = st.sidebar.multiselect("Area", area_options, default=["All"])
    if "All" in selected_areas or not selected_areas:
        selected_areas = areas
else:
    selected_areas = []

def get_range(col_name, default_max=1000.0):
    """Creates a range slider for a specified column."""
    if col_name and d[col_name].notna().any():
        col_min, col_max = float(d[col_name].min()), float(d[col_name].max())
        # Prevent max == min error
        if col_max == col_min: col_max += 1
    else:
        col_min, col_max = 0.0, default_max
    return st.sidebar.slider(f"{col_name} range" if col_name else "Range", col_min, col_max, (col_min, col_max))

price_range = get_range(PRICE_COL)
sales_range = get_range(INCOME_COL)

# Apply filters
df_filtered = d.copy()
if AREA_COL and selected_areas:
    df_filtered = df_filtered[df_filtered[AREA_COL].isin(selected_areas)]
if PRICE_COL:
    df_filtered = df_filtered[df_filtered[PRICE_COL].between(*price_range)]
if INCOME_COL:
    df_filtered = df_filtered[df_filtered[INCOME_COL].between(*sales_range)]
    
# Skip the rest if no data remains
if df_filtered.empty:
    st.error("No data matches the selected filters and mappings.")
    st.stop()


# ---------------------------------------------------
# MAIN TABS
# ---------------------------------------------------
tabs = st.tabs([
    "Overview", "City Analysis", "Customer Insights",
    "Host Performance", "Prediction", "Raw Data"
])

# ---------------------------------------------------
# 1. OVERVIEW
# ---------------------------------------------------
with tabs[0]:
    st.header("Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Listings (filtered)", len(df_filtered))
    col2.metric("Avg Price", f"${df_filtered[PRICE_COL].mean():,.2f}" if PRICE_COL else "N/A")
    col3.metric("Avg Sales", f"${df_filtered[INCOME_COL].mean():,.2f}" if INCOME_COL else "N/A")
    col4.metric("Avg Reviewers", f"{df_filtered[REVIEWERS_COL].mean():.1f}" if REVIEWERS_COL else "N/A")

    if PRICE_COL:
        st.markdown("### Price Distribution")
        st.altair_chart(
            alt.Chart(df_filtered).mark_bar().encode(
                alt.X(PRICE_COL, bin=alt.Bin(maxbins=60), title="Price"),
                y="count()"
            ).properties(height=300),
            use_container_width=True
        )

    if INCOME_COL:
        st.markdown("### Income (Sales) Distribution")
        st.altair_chart(
            alt.Chart(df_filtered).mark_bar().encode(
                alt.X(INCOME_COL, bin=alt.Bin(maxbins=60), title="Sales"),
                y="count()"
            ).properties(height=300),
            use_container_width=True
        )

    if CITY_COL:
        st.markdown("### Listings per City (Top 25)")
        city_counts = df_filtered[CITY_COL].value_counts().head(25).reset_index()
        city_counts.columns = ["City", "Listing_Count"]
        st.altair_chart(
            alt.Chart(city_counts).mark_bar().encode(
                x=alt.X("City", sort='-y'),
                y="Listing_Count",
                tooltip=["City", "Listing_Count"]
            ).properties(height=300),
            use_container_width=True
        )

# ---------------------------------------------------
# 2. CITY ANALYSIS
# ---------------------------------------------------
with tabs[1]:
    st.header("City Analysis")
    
    REQUIRED_CITY_COLS = [CITY_COL, PRICE_COL, INCOME_COL, REVIEWERS_COL]
    if all(REQUIRED_CITY_COLS):
        # NOTE: Fallback to 'size' removed, requiring all mapped columns for clarity
        agg = df_filtered.groupby(CITY_COL).agg({
            PRICE_COL: "mean", 
            INCOME_COL: "sum", 
            REVIEWERS_COL: "mean"
        }).reset_index().rename(columns={
            PRICE_COL: "avg_price",
            INCOME_COL: "total_sales",
            REVIEWERS_COL: "avg_reviewers"
        })

        st.dataframe(agg.sort_values(by="total_sales", ascending=False).head(100))

        st.markdown("### Bubble Chart: Avg Price vs Avg Reviewers (size = Sales)")
        st.altair_chart(
            alt.Chart(agg).mark_circle().encode(
                x="avg_price", y="avg_reviewers", size="total_sales", color=CITY_COL,
                tooltip=[CITY_COL, "avg_price", "avg_reviewers", "total_sales"]
            ).interactive().properties(height=450),
            use_container_width=True
        )
    else:
        st.info("To view City Analysis, please map the **City**, **Price**, **Income**, and **Reviewers** columns in the sidebar.")

# ---------------------------------------------------
# 3. CUSTOMER INSIGHTS
# ---------------------------------------------------
with tabs[2]:
    st.header("Customer Insights")
    if PRICE_COL and REVIEWERS_COL:
        st.markdown("### Price vs Reviewers vs Sales")
        st.altair_chart(
            alt.Chart(df_filtered).mark_circle().encode(
                x=PRICE_COL, y=REVIEWERS_COL,
                size=INCOME_COL if INCOME_COL else alt.value(40),
                tooltip=[CITY_COL or "index", PRICE_COL, REVIEWERS_COL, INCOME_COL or ""]
            ).interactive().properties(height=450),
            use_container_width=True
        )

    if REVIEWERS_COL and INCOME_COL:
        corr = df_filtered[[REVIEWERS_COL, INCOME_COL]].dropna()
        if not corr.empty:
            st.info(f"Correlation (Reviewers vs Sales): **{corr[REVIEWERS_COL].corr(corr[INCOME_COL]):.3f}**")
        else:
            st.info("Not enough non-missing data to calculate correlation.")

# ---------------------------------------------------
# 4. HOST PERFORMANCE
# ---------------------------------------------------
with tabs[3]:
    st.header("Host Performance")
    if HOST_RESP_COL:
        st.subheader("Host Response Rate")
        st.altair_chart(
            alt.Chart(df_filtered).mark_bar().encode(
                alt.X(HOST_RESP_COL, bin=alt.Bin(maxbins=30), title="Response Rate (%)"),
                y="count()"
            ).properties(height=300),
            use_container_width=True
        )
    if HOST_ACCEPT_COL:
        st.subheader("Host Acceptance Rate")
        st.altair_chart(
            alt.Chart(df_filtered).mark_bar().encode(
                alt.X(HOST_ACCEPT_COL, bin=alt.Bin(maxbins=30), title="Acceptance Rate (%)"),
                y="count()"
            ).properties(height=300),
            use_container_width=True
        )

# ---------------------------------------------------
# 5. PREDICTION
# ---------------------------------------------------
with tabs[4]:
    st.header("Predict Sales (Income)")
    if INCOME_COL:
        model_df = df_filtered.select_dtypes(include=[np.number]).copy()
        
        if INCOME_COL not in model_df.columns:
            st.warning("Income column is not numeric after cleaning.")
        else:
            # 🚨 FIX: Drop rows where the target (Income) is missing
            model_df.dropna(subset=[INCOME_COL], inplace=True) 
            
            if model_df.shape[0] < 10:
                st.warning("Not enough data points remaining for a robust prediction after filtering.")
            else:
                X = model_df.drop(columns=[INCOME_COL], errors="ignore")
                y = model_df[INCOME_COL]
                
                # 🚨 FIX: Impute features (X) with the mean of the column
                X.fillna(X.mean(), inplace=True) 
                
                if X.shape[1] >= 2:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
                    
                    with st.spinner("Training Random Forest Regressor..."):
                        model.fit(X_train, y_train)
                        preds = model.predict(X_test)
                        
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("MAE (Mean Absolute Error)", f"{mean_absolute_error(y_test, preds):,.2f}")
                    col_m2.metric("R² (Coefficient of Determination)", f"{r2_score(y_test, preds):.3f}")
                    
                else:
                    st.warning("Not enough numeric features remaining for prediction (need at least 2).")
    else:
        st.warning("No income column found for prediction.")

# ---------------------------------------------------
# 6. RAW DATA
# ---------------------------------------------------
with tabs[5]:
    st.header("Raw Data (Filtered)")
    st.write(f"Filtered rows: {len(df_filtered)}")
    
    # Use st.cache_data for faster CSV export
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    csv = convert_df_to_csv(df_filtered)

    st.download_button("Download Filtered CSV",
        csv,
        "filtered_listings.csv",
        "text/csv"
    )
    st.dataframe(df_filtered.head(1000))

st.caption(" Educonnect Rwanda - Hotel analysis project.")
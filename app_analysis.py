import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
st.set_page_config(page_title="🏨 Airbnb Hotel - Upgraded Dashboard", layout="wide")
st.title("🏨 Airbnb Hotel Analysis — Upgraded Dashboard")

# ✅ Preferred data source: GitHub raw file
GITHUB_CSV_URL = "https://raw.githubusercontent.com/nshibone/educonnect_final_project/master/Airbnb_site_hotel%20new.csv"


# ---------------------------------------------------
# DATA LOADING & CLEANING
# ---------------------------------------------------
@st.cache_data
def load_data():
    """Load CSV from GitHub (preferred) or fallback to local path."""
    try:
        st.info("Loading dataset from GitHub...")
        df = pd.read_csv(GITHUB_CSV_URL, low_memory=False)
        st.success("✅ Data loaded successfully from GitHub!")
        return df
    except Exception as e:
        st.warning(f"⚠️ Could not load from GitHub: {e}. Trying local path...")
        try:
            df = pd.read_csv(LOCAL_PATH, low_memory=False)
            st.success("✅ Data loaded successfully from local path.")
            return df
        except FileNotFoundError:
            st.error("❌ No valid dataset found. Please ensure the CSV file exists or the GitHub link is accessible.")
            st.stop()

def clean_price(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float)): return x
    s = str(x).replace("$", "").replace(",", "").replace("USD", "").strip()
    try:
        return float(s)
    except:
        try:
            return float(s.replace(".", "").replace(",", "."))
        except:
            return np.nan

def clean_percent(x):
    if pd.isna(x): return np.nan
    s = str(x).replace("%", "").replace(",", ".").strip()
    try:
        return float(s)
    except:
        return np.nan

def safe_column(df, name, alt_names=[]):
    for n in [name] + alt_names:
        if n in df.columns:
            return n
    return None

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
df = load_data()

# ---------------------------------------------------
# SIDEBAR COLUMN MAPPING
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

def select_with_suggestion(label, suggest):
    idx = cols.index(suggest) if suggest in cols else 0
    return st.sidebar.selectbox(label, cols, index=idx)

CITY_COL = select_with_suggestion("City column", suggestions["City"])
AREA_COL = select_with_suggestion("Area column", suggestions["Area"])
PRICE_COL = select_with_suggestion("Price column", suggestions["Price"])
INCOME_COL = select_with_suggestion("Income / Sales column", suggestions["Income"])
REVIEWERS_COL = select_with_suggestion("Reviewers column", suggestions["Reviewers"])
HOST_RESP_COL = select_with_suggestion("Host Response Rate column", suggestions["Host Response"])
HOST_ACCEPT_COL = select_with_suggestion("Host Acceptance Rate column", suggestions["Host Acceptance"])

def none_or_value(val): return None if val == "None" else val
CITY_COL, AREA_COL, PRICE_COL, INCOME_COL, REVIEWERS_COL, HOST_RESP_COL, HOST_ACCEPT_COL = map(
    none_or_value, [CITY_COL, AREA_COL, PRICE_COL, INCOME_COL, REVIEWERS_COL, HOST_RESP_COL, HOST_ACCEPT_COL]
)

st.sidebar.markdown("### 🔎 Current mappings")
st.sidebar.write({
    "City": CITY_COL,
    "Area": AREA_COL,
    "Price": PRICE_COL,
    "Income": INCOME_COL,
    "Reviewers": REVIEWERS_COL,
    "Host Response": HOST_RESP_COL,
    "Host Acceptance": HOST_ACCEPT_COL,
})

# ---------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------
d = df.copy()
if PRICE_COL: d[PRICE_COL] = d[PRICE_COL].apply(clean_price)
if INCOME_COL: d[INCOME_COL] = d[INCOME_COL].apply(clean_price)
if REVIEWERS_COL: d[REVIEWERS_COL] = pd.to_numeric(d[REVIEWERS_COL], errors="coerce")
if HOST_RESP_COL: d[HOST_RESP_COL] = d[HOST_RESP_COL].apply(clean_percent)
if HOST_ACCEPT_COL: d[HOST_ACCEPT_COL] = d[HOST_ACCEPT_COL].apply(clean_percent)
if PRICE_COL and REVIEWERS_COL:
    d["price_per_reviewer"] = d[PRICE_COL] / (d[REVIEWERS_COL].replace({0: np.nan}))

# ---------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------
st.sidebar.header("🔍 Filters")

if AREA_COL:
    areas = sorted(d[AREA_COL].dropna().unique().tolist())
    area_options = ["All"] + areas
    selected_areas = st.sidebar.multiselect("Area", area_options, default=["All"])
    if "All" in selected_areas or not selected_areas:
        selected_areas = areas
else:
    selected_areas = []

def get_range(col_name, default_max=1000.0):
    if col_name and d[col_name].notna().any():
        col_min, col_max = float(d[col_name].min()), float(d[col_name].max())
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
    if CITY_COL:
        agg_cols = {}
        if PRICE_COL: agg_cols[PRICE_COL] = "mean"
        if INCOME_COL: agg_cols[INCOME_COL] = "sum"
        if REVIEWERS_COL: agg_cols[REVIEWERS_COL] = "mean"

        if agg_cols:
            agg = df_filtered.groupby(CITY_COL).agg(agg_cols).reset_index().rename(columns={
                PRICE_COL: "avg_price",
                INCOME_COL: "total_sales",
                REVIEWERS_COL: "avg_reviewers"
            })

            sort_by_col = "total_sales" if INCOME_COL else "avg_price"
            st.dataframe(agg.sort_values(by=sort_by_col, ascending=False).head(100))

            if PRICE_COL and REVIEWERS_COL and INCOME_COL:
                st.markdown("### Bubble Chart: Avg Price vs Avg Reviewers (size = Sales)")
                st.altair_chart(
                    alt.Chart(agg).mark_circle().encode(
                        x="avg_price", y="avg_reviewers", size="total_sales", color=CITY_COL,
                        tooltip=[CITY_COL, "avg_price", "avg_reviewers", "total_sales"]
                    ).interactive().properties(height=450),
                    use_container_width=True
                )
        else:
            st.warning("Please select numeric columns for City Analysis.")

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
        if len(corr) > 1:
            st.info(f"Correlation (Reviewers vs Sales): **{corr[REVIEWERS_COL].corr(corr[INCOME_COL]):.3f}**")
        else:
            st.info("Not enough data to calculate correlation after dropping NaNs.")

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
        model_df = df_filtered.select_dtypes(include=[np.number]).dropna(axis=1, how="all")
        if INCOME_COL not in model_df.columns or not model_df[INCOME_COL].notna().any():
            st.warning("Income column missing or not numeric.")
        else:
            X = model_df.drop(columns=[INCOME_COL], errors="ignore").fillna(0)
            y = model_df[INCOME_COL].fillna(model_df[INCOME_COL].median())

            if X.shape[1] >= 1 and X.shape[0] >= 10:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                col_m, col_r = st.columns(2)
                col_m.metric("MAE (Mean Absolute Error)", f"{mean_absolute_error(y_test, preds):,.2f}")
                col_r.metric("R² (Coefficient of Determination)", f"{r2_score(y_test, preds):.3f}")
            else:
                st.warning("Not enough numeric features or rows for prediction.")
    else:
        st.warning("No income column found for prediction.")

# ---------------------------------------------------
# 6. RAW DATA
# ---------------------------------------------------
with tabs[5]:
    st.header("Raw Data (Filtered)")
    st.write(f"Filtered rows: {len(df_filtered)}")
    st.download_button(
        "Download Filtered CSV",
        df_filtered.to_csv(index=False).encode("utf-8"),
        "filtered_listings.csv"
    )
    st.dataframe(df_filtered.head(1000))

st.caption(" Educonnect Rwanda - Hotel analysis project.")

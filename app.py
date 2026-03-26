import streamlit as st
import pandas as pd

# Import your modules
from data_processing import run_pipeline
import visualizations as viz

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

# =========================
# THEME TOGGLE
# =========================
theme = st.sidebar.toggle("🌙 Dark Mode", value=False)

if theme:
    st.markdown(
        """
        <style>
        body { background-color: #0E1117; color: white; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# LOAD DATA (CACHED)
# =========================
@st.cache_data
def load_data():
    return run_pipeline()

data = load_data()

df = data["df"]
customer_metrics = data["customer_metrics"]
product_metrics = data["product_metrics"]
country_metrics = data["country_metrics"]
rfm = data["rfm"]
segment_summary = data["segment_summary"]
retention = data["retention_pct"]

# =========================
# SIDEBAR FILTERS
# =========================
st.sidebar.header("Filters")

# Date filter
min_date = df["InvoiceDate"].min()
max_date = df["InvoiceDate"].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date]
)

# Country filter
countries = st.sidebar.multiselect(
    "Select Country",
    options=df["Country"].unique(),
    default=df["Country"].unique()
)

# Segment filter
segments = st.sidebar.multiselect(
    "Select Segment",
    options=rfm["Segment"].unique(),
    default=rfm["Segment"].unique()
)

# =========================
# APPLY FILTERS
# =========================
df_filtered = df[
    (df["Country"].isin(countries)) &
    (df["InvoiceDate"].dt.date >= date_range[0]) &
    (df["InvoiceDate"].dt.date <= date_range[1])
]

rfm_filtered = rfm[rfm["Segment"].isin(segments)]

# =========================
# NAVIGATION
# =========================
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Customer Analytics",
        "Product Analytics",
        "RFM Segmentation",
        "Cohort Analysis"
    ]
)

# =========================
# KPI FUNCTION
# =========================
def kpi_card(title, value):
    st.markdown(f"""
        <div style="padding:15px;border-radius:10px;background:#1f2937">
            <h4>{title}</h4>
            <h2>{value}</h2>
        </div>
    """, unsafe_allow_html=True)

# =========================
# OVERVIEW PAGE
# =========================
if page == "Overview":
    st.title("Overview Dashboard")

    total_revenue = df_filtered["TotalPrice"].sum()
    total_orders = df_filtered["InvoiceNo"].nunique()
    total_customers = df_filtered["CustomerID"].nunique()
    avg_order = total_revenue / total_orders if total_orders != 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kpi_card("Total Revenue", f"${total_revenue:,.0f}")
    with col2:
        kpi_card("Total Orders", total_orders)
    with col3:
        kpi_card("Customers", total_customers)
    with col4:
        kpi_card("Avg Order Value", f"${avg_order:,.2f}")

    st.subheader("Revenue Trend")
    st.pyplot(viz.plot_monthly_sales(df_filtered))

    st.subheader("Country Analysis")
    st.pyplot(viz.plot_country_analysis(df_filtered))

    st.subheader("Top Products")
    st.pyplot(viz.plot_top_products(df_filtered))


elif page == "Customer Analytics":
    st.title("Customer Analytics")

    st.subheader("Customer Retention and Churn")
    st.pyplot(viz.plot_retention_churn(df_filtered, rfm_filtered))

    st.subheader("Order Value Distribution")
    st.pyplot(viz.plot_order_value_distribution(df_filtered))


elif page == "Product Analytics":
    st.title("Product Analytics")

    st.subheader("Top Products by Revenue")
    st.pyplot(viz.plot_top_products(df_filtered))

    st.subheader("Basket Size Analysis")
    st.pyplot(viz.plot_basket_size(df_filtered))


elif page == "RFM Segmentation":
    st.title("RFM Segmentation")

    st.subheader("RFM Overview")
    st.pyplot(viz.plot_rfm_overview(rfm_filtered))

    st.subheader("Segment Summary")
    st.dataframe(segment_summary)


elif page == "Cohort Analysis":
    st.title("Cohort Analysis")

    st.subheader("Retention Heatmap")
    st.pyplot(viz.plot_cohort_retention(retention))


st.sidebar.download_button(
    label="Download Data",
    data=df_filtered.to_csv(index=False),
    file_name="filtered_data.csv",
    mime="text/csv"
)
print("hello world")
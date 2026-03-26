import pandas as pd
import numpy as np
def load_data(filepath: str = None) -> pd.DataFrame:
    # If local file is provided, use it
    if filepath:
        return pd.read_csv(filepath, encoding="latin-1", on_bad_lines="skip")
    
    # Otherwise load from Google Drive
    url = "https://drive.google.com/uc?id=1YKr-fw8Arb3iVHhdYrJxqQRakL-e--Fl"
    df = pd.read_csv(url, encoding="latin-1", on_bad_lines="skip")
    
    return df
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Remove nulls
    df = df.dropna(subset=["CustomerID", "Description"])

    # Drop duplicates and fix types
    df = df.drop_duplicates()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Remove cancellations
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Filter invalid rows
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # IQR-based outlier removal on Quantity
    Q1 = df["Quantity"].quantile(0.25)
    Q3 = df["Quantity"].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df["Quantity"] >= lower_bound) & (df["Quantity"] <= upper_bound)]

    return df.reset_index(drop=True)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TotalPrice
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # Date columns
    df["Year"] = df["InvoiceDate"].dt.year
    df["Month"] = df["InvoiceDate"].dt.month
    df["Day"] = df["InvoiceDate"].dt.day
    df["Hour"] = df["InvoiceDate"].dt.hour

    return df


def get_customer_metrics(df: pd.DataFrame) -> pd.DataFrame:
    customer_revenue = (
        df.groupby("CustomerID")["TotalPrice"].sum().reset_index()
    )
    customer_revenue.columns = ["CustomerID", "TotalRevenue"]

    orders_per_customer = (
        df.groupby("CustomerID")["InvoiceNo"]
        .nunique()
        .reset_index()
        .rename(columns={"InvoiceNo": "OrderCount"})
    )

    customer_metrics = customer_revenue.merge(orders_per_customer, on="CustomerID")
    customer_metrics["AvgOrderValue"] = (
        customer_metrics["TotalRevenue"] / customer_metrics["OrderCount"]
    )
    customer_metrics["IsRepeat"] = customer_metrics["OrderCount"] > 1

    return customer_metrics


def get_product_metrics(df: pd.DataFrame) -> pd.DataFrame:
    product_metrics = (
        df.groupby("Description")
        .agg(TotalQuantity=("Quantity", "sum"), TotalRevenue=("TotalPrice", "sum"))
        .sort_values("TotalRevenue", ascending=False)
        .reset_index()
    )
    return product_metrics
def get_country_metrics(df: pd.DataFrame) -> pd.DataFrame:
    country_metrics = (
        df.groupby("Country")
        .agg(
            TotalRevenue=("TotalPrice", "sum"),
            OrderCount=("InvoiceNo", "nunique"),
        )
        .sort_values("TotalRevenue", ascending=False)
        .reset_index()
    )
    return country_metrics
def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )

    rfm["R"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["M"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

    rfm["RFM_Score"] = rfm[["R", "F", "M"]].astype(int).sum(axis=1)

    def _segment(score: int) -> str:
        if score >= 13:
            return "Champions"
        elif score >= 10:
            return "Loyal Customers"
        elif score >= 7:
            return "Potential Loyalists"
        elif score >= 5:
            return "At Risk"
        else:
            return "Lost"

    rfm["Segment"] = rfm["RFM_Score"].apply(_segment)

    return rfm


def get_segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rfm.groupby("Segment")
        .agg(
            Recency=("Recency", "mean"),
            Frequency=("Frequency", "mean"),
            Monetary=("Monetary", "mean"),
            CustomerCount=("RFM_Score", "count"),
        )
        .round(2)
        .sort_values("CustomerCount", ascending=False)
    )
    return summary

def prepare_cohort_data(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M")

    # Step 1 – cohort month (first purchase month per customer)
    cohort_month = (
        df.groupby("CustomerID")["InvoiceMonth"]
        .min()
        .reset_index()
        .rename(columns={"InvoiceMonth": "CohortMonth"})
    )
    df = df.merge(cohort_month, on="CustomerID")

    # Step 2 – cohort index
    df["CohortIndex"] = (
        df["InvoiceMonth"].dt.start_time.dt.to_period("M").astype(int)
        - df["CohortMonth"].dt.start_time.dt.to_period("M").astype(int)
    )

    # Step 3 – retention table (unique customers per cohort × index)
    cohort_pivot = (
        df.groupby(["CohortMonth", "CohortIndex"])["CustomerID"]
        .nunique()
        .reset_index()
        .pivot(index="CohortMonth", columns="CohortIndex", values="CustomerID")
    )

    # Step 4 – percentage retention (relative to cohort size at index 0)
    retention_pct = cohort_pivot.divide(cohort_pivot.iloc[:, 0], axis=0).round(3)

    return {
        "cohort_data": df,
        "retention_table": cohort_pivot,
        "retention_pct": retention_pct,
    }


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================

def run_pipeline(filepath: str = None) -> dict:
    df = load_data(filepath)
    df = clean_data(df)
    df = engineer_features(df)

    customer_metrics = get_customer_metrics(df)
    product_metrics = get_product_metrics(df)
    country_metrics = get_country_metrics(df)

    rfm = calculate_rfm(df)
    segment_summary = get_segment_summary(rfm)

    cohort_results = prepare_cohort_data(df)

    return {
        "df": df,
        "customer_metrics": customer_metrics,
        "product_metrics": product_metrics,
        "country_metrics": country_metrics,
        "rfm": rfm,
        "segment_summary": segment_summary,
        "cohort_data": cohort_results["cohort_data"],
        "retention_table": cohort_results["retention_table"],
        "retention_pct": cohort_results["retention_pct"],
    }
print("hellowordl")
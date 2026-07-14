import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Pandas-compatible month-end frequency
try:
    pd.date_range("2020-01-01", periods=1, freq="ME")
    MONTH_END = "ME"   # Newer pandas
except ValueError:
    MONTH_END = "M"    # Older pandas


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("End-to-End Sales Forecasting & Demand Intelligence System")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")

    df["Order Date"] = pd.to_datetime(
        df["Order Date"],
        format="%d/%m/%Y"
    )

    return df

df = load_data()

# ============================================================
# HELPER FUNCTION: SEASON
# ============================================================

def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Spring"

    elif month in [6, 7, 8]:
        return "Summer"

    else:
        return "Fall"


# ============================================================
# PREPARE XGBOOST FEATURES
# ============================================================

def prepare_xgb_data(segment_data):

    # Convert order-level data to monthly sales
    monthly = (
        segment_data
        .set_index("Order Date")["Sales"]
        .resample(MONTH_END)
        .sum()
        .reset_index()
    )

    # Lag features
    monthly["Lag_1"] = monthly["Sales"].shift(1)
    monthly["Lag_2"] = monthly["Sales"].shift(2)
    monthly["Lag_3"] = monthly["Sales"].shift(3)

    # Rolling mean using previous 3 months
    monthly["Rolling_Mean_3"] = (
        monthly["Sales"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    # Time features
    monthly["Month"] = monthly["Order Date"].dt.month
    monthly["Quarter"] = monthly["Order Date"].dt.quarter

    monthly["Season"] = (
        monthly["Month"]
        .apply(get_season)
    )

    # Encode season
    monthly = pd.get_dummies(
        monthly,
        columns=["Season"],
        dtype=int
    )

    return monthly


# ============================================================
# XGBOOST FORECAST FUNCTION
# ============================================================

def forecast_segment(segment_data, horizon):

    monthly = prepare_xgb_data(segment_data)

    model_data = monthly.dropna().copy()

    X = model_data.drop(
        columns=["Order Date", "Sales"]
    )

    y = model_data["Sales"]


    # --------------------------------------------------------
    # MODEL EVALUATION
    # Last 3 months used as test data
    # --------------------------------------------------------

    X_train = X.iloc[:-3]
    X_test = X.iloc[-3:]

    y_train = y.iloc[:-3]
    y_test = y.iloc[-3:]


    evaluation_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        objective="reg:squarederror"
    )

    evaluation_model.fit(
        X_train,
        y_train
    )

    test_predictions = (
        evaluation_model.predict(X_test)
    )


    # Evaluation metrics
    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            test_predictions
        )
    )


    # --------------------------------------------------------
    # FINAL MODEL
    # Train using all available data
    # --------------------------------------------------------

    final_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        objective="reg:squarederror"
    )

    final_model.fit(X, y)


    # Historical sales
    sales_history = (
        monthly["Sales"].tolist()
    )


    # Future dates
    future_dates = pd.date_range(
        start=(
            monthly["Order Date"].max()
            + pd.offsets.MonthEnd(1)
        ),
        periods=horizon,
        freq=MONTH_END
    )


    future_predictions = []


    # Recursive forecasting
    for date in future_dates:

        month = date.month
        quarter = date.quarter
        season = get_season(month)


        future_row = pd.DataFrame({

            "Lag_1": [
                sales_history[-1]
            ],

            "Lag_2": [
                sales_history[-2]
            ],

            "Lag_3": [
                sales_history[-3]
            ],

            "Rolling_Mean_3": [
                np.mean(
                    sales_history[-3:]
                )
            ],

            "Month": [
                month
            ],

            "Quarter": [
                quarter
            ],

            "Season_Fall": [
                1 if season == "Fall" else 0
            ],

            "Season_Spring": [
                1 if season == "Spring" else 0
            ],

            "Season_Summer": [
                1 if season == "Summer" else 0
            ],

            "Season_Winter": [
                1 if season == "Winter" else 0
            ]

        })


        # Match training feature columns
        future_row = future_row.reindex(
            columns=X.columns,
            fill_value=0
        )


        prediction = (
            final_model.predict(
                future_row
            )[0]
        )


        future_predictions.append(
            prediction
        )


        # Prediction becomes lag for next month
        sales_history.append(
            prediction
        )


    forecast_df = pd.DataFrame({

        "Date":
            future_dates,

        "Forecasted Sales":
            future_predictions

    })


    return (
        forecast_df,
        mae,
        rmse
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(

    "Select Page",

    [
        "Sales Overview Dashboard",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)


# ============================================================
# PAGE 1 — SALES OVERVIEW
# ============================================================

if page == "Sales Overview Dashboard":

    st.header("📊 Sales Overview Dashboard")


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    total_sales = df["Sales"].sum()

    total_orders = (
        df["Order ID"]
        .nunique()
    )

    total_customers = (
        df["Customer ID"]
        .nunique()
    )


    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Total Sales",
        f"${total_sales:,.2f}"
    )


    col2.metric(
        "Total Orders",
        f"{total_orders:,}"
    )


    col3.metric(
        "Total Customers",
        f"{total_customers:,}"
    )


    st.divider()


    # --------------------------------------------------------
    # TOTAL SALES BY YEAR
    # --------------------------------------------------------

    st.subheader("Total Sales by Year")


    yearly_sales = (df.groupby(df["Order Date"].dt.year)["Sales"].sum().reset_index())


    yearly_sales.columns = ["Year","Sales"]


    st.bar_chart(yearly_sales,x="Year",y="Sales")


    # --------------------------------------------------------
    # MONTHLY SALES TREND
    # --------------------------------------------------------

    st.subheader("Monthly Sales Trend")


    monthly_sales = (

        df.set_index(
            "Order Date"
        )["Sales"]

        .resample(MONTH_END)

        .sum()

    )


    st.line_chart(
        monthly_sales
    )


    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.subheader(
        "Sales by Region and Category"
    )


    selected_regions = st.multiselect(

        "Select Region",

        options=sorted(
            df["Region"].unique()
        ),

        default=sorted(
            df["Region"].unique()
        )

    )


    selected_categories = st.multiselect(

        "Select Category",

        options=sorted(
            df["Category"].unique()
        ),

        default=sorted(
            df["Category"].unique()
        )

    )


    filtered_df = df[

        df["Region"].isin(
            selected_regions
        )

        &

        df["Category"].isin(
            selected_categories
        )

    ]


    filtered_sales = (

        filtered_df

        .groupby(
            [
                "Region",
                "Category"
            ]
        )["Sales"]

        .sum()

        .reset_index()

    )


    if not filtered_sales.empty:

        st.bar_chart(

            filtered_sales,

            x="Region",

            y="Sales",

            color="Category"

        )

    else:

        st.warning(
            "Please select at least one region and category."
        )


# ============================================================
# PAGE 2 — FORECAST EXPLORER
# ============================================================

elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")


    st.write("Forecasting model: XGBoost Regressor")


    # Select Category or Region
    segment_type = st.selectbox("Select Forecast Type",["Category","Region"])


    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

    if segment_type == "Category":
        selected_segment = st.selectbox("Select Category",sorted(df["Category"].unique()))


        segment_data = df[df["Category"] == selected_segment].copy()


    # --------------------------------------------------------
    # REGION
    # --------------------------------------------------------

    else:

        selected_segment = st.selectbox("Select Region",sorted(df["Region"].unique()))


        segment_data = df[ df["Region"]== selected_segment].copy()


    # Forecast horizon
    horizon = st.slider(

        "Select Forecast Horizon (Months)",

        min_value=1,

        max_value=3,

        value=3,

        step=1

    )


    # Run forecast
    forecast_df, mae, rmse = (forecast_segment(segment_data,horizon))


    # --------------------------------------------------------
    # FORECAST CHART
    # --------------------------------------------------------

    st.subheader(

        f"{horizon}-Month Sales Forecast "
        f"for {selected_segment}"

    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    ax.plot(

        forecast_df["Date"],

        forecast_df[ "Forecasted Sales"],

        marker="o",

        label="Forecasted Sales"

    )


    ax.set_title(

        f"XGBoost Forecast — "
        f"{selected_segment}"

    )


    ax.set_xlabel("Date")


    ax.set_ylabel("Forecasted Sales")


    ax.legend()

    ax.grid(True)


    st.pyplot(fig)


    # --------------------------------------------------------
    # FORECAST TABLE
    # --------------------------------------------------------

    st.subheader("Forecast Values")


    display_forecast = (forecast_df.copy())


    display_forecast["Forecasted Sales"] = (display_forecast["Forecasted Sales"].round(2))


    st.dataframe(display_forecast,use_container_width=True)


    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    st.subheader("Model Performance")


    col1, col2 = st.columns(2)


    col1.metric("MAE",f"{mae:,.2f}")


    col2.metric("RMSE",f"{rmse:,.2f}")


# ============================================================
# PAGE 3 — ANOMALY REPORT
# ============================================================

elif page == "Anomaly Report":

    st.header("🚨 Weekly Sales Anomaly Report")


    # --------------------------------------------------------
    # WEEKLY SALES
    # --------------------------------------------------------

    weekly_sales = (

        df.set_index("Order Date")["Sales"]
        .resample("W")
        .sum()
        .reset_index()

    )


    # --------------------------------------------------------
    # ISOLATION FOREST
    # --------------------------------------------------------

    isolation_model = IsolationForest(contamination=0.05,random_state=42)


    weekly_sales["IF_Anomaly"] = (isolation_model.fit_predict(weekly_sales[["Sales"]]))


    weekly_sales["IF_Flag"] = (weekly_sales["IF_Anomaly"]== -1)


    if_anomalies = weekly_sales[weekly_sales["IF_Flag"]]


    # --------------------------------------------------------
    # ROLLING Z-SCORE
    # --------------------------------------------------------

    window = 12


    weekly_sales["Rolling_Mean"] = (

        weekly_sales["Sales"]

        .rolling(
            window=window,
            min_periods=window
        )

        .mean()

    )


    weekly_sales["Rolling_Std"] = (

        weekly_sales["Sales"].rolling(
            window=window,
            min_periods=window
        )

        .std()

    )


    weekly_sales["Z_Score"] = ( (weekly_sales["Sales"] - weekly_sales["Rolling_Mean"])/ weekly_sales["Rolling_Std"])


    weekly_sales["Z_Anomaly"] = ( weekly_sales["Z_Score"].abs()> 2)


    z_anomalies = weekly_sales[weekly_sales["Z_Anomaly"]]


    # --------------------------------------------------------
    # ANOMALY COMPARISON CHART
    # --------------------------------------------------------

    st.subheader("Anomaly Detection Comparison")


    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(weekly_sales["Order Date"],weekly_sales["Sales"],label="Weekly Sales")


    ax.scatter( if_anomalies["Order Date"],

        if_anomalies["Sales"],

        marker="o",

        s=70,

        label="Isolation Forest"

    )


    ax.scatter(
        z_anomalies["Order Date"],
        z_anomalies["Sales"],

        marker="X",

        s=80,

        label="Z-Score"

    )


    ax.set_title( "Weekly Sales Anomaly Detection")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weekly Sales")


    ax.legend()

    ax.grid(True)


    st.pyplot(fig)


    # --------------------------------------------------------
    # ANOMALY TABLE
    # --------------------------------------------------------

    st.subheader("Detected Anomaly Dates")


    anomaly_table = weekly_sales[weekly_sales["IF_Flag"]|weekly_sales["Z_Anomaly"]][["Order Date","Sales","IF_Flag","Z_Anomaly"]].copy()


    anomaly_table.columns = [ "Date", "Weekly Sales", "Isolation Forest","Z-Score"]


    st.dataframe(anomaly_table,use_container_width=True)


# ============================================================
# PAGE 4 — PRODUCT DEMAND SEGMENTS
# ============================================================

elif page == "Product Demand Segments":

    st.header("📦 Product Demand Segmentation")


    # --------------------------------------------------------
    # TOTAL SALES AND AVERAGE ORDER VALUE
    # --------------------------------------------------------

    subcat_features = (df.groupby("Sub-Category").agg(Total_Sales=("Sales","sum"), Average_Order_Value=("Sales","mean")).reset_index())


    # --------------------------------------------------------
    # MONTHLY SALES VOLATILITY
    # --------------------------------------------------------

    monthly_subcat = (df.groupby(["Sub-Category",pd.Grouper(key="Order Date",freq="M")])["Sales"].sum().reset_index())
    volatility = (monthly_subcat.groupby("Sub-Category")["Sales"].std().reset_index(name="Sales_Volatility"))


    # --------------------------------------------------------
    # YEAR-OVER-YEAR GROWTH
    # --------------------------------------------------------

    yearly_subcat = (df.groupby(["Sub-Category",df["Order Date"].dt.year.rename("Year")])["Sales"].sum().reset_index())


    yearly_subcat["YoY_Growth"] = (yearly_subcat.groupby("Sub-Category")["Sales"].pct_change()* 100)


    average_growth = (yearly_subcat.groupby("Sub-Category")["YoY_Growth"].mean().reset_index(name="Sales_Growth_Rate"))


    # --------------------------------------------------------
    # MERGE FEATURES
    # --------------------------------------------------------

    subcat_features = (subcat_features.merge(volatility,on="Sub-Category").merge(average_growth,on="Sub-Category"))


    # Features used for clustering
    feature_columns = ["Total_Sales","Sales_Growth_Rate","Sales_Volatility","Average_Order_Value"]


    # --------------------------------------------------------
    # SCALE FEATURES
    # --------------------------------------------------------

    scaler = StandardScaler()


    X_scaled = scaler.fit_transform(subcat_features[feature_columns])


    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    kmeans = KMeans(n_clusters=4,random_state=42,n_init=10)


    subcat_features["Cluster"] = (kmeans.fit_predict(X_scaled))


    # --------------------------------------------------------
    # CLUSTER LABELS
    # --------------------------------------------------------

    cluster_labels = {

        0:"High Volume, Stable Demand",

        1:"Low Volume, Stable Demand",

        2:"Rapidly Growing Demand",

        3:"High-Value, Volatile Demand"

    }


    subcat_features[
        "Demand Cluster"
    ] = (

        subcat_features[
            "Cluster"
        ]

        .map(
            cluster_labels
        )

    )


    # --------------------------------------------------------
    # PCA
    # --------------------------------------------------------

    pca = PCA(
        n_components=2
    )


    X_pca = pca.fit_transform(
        X_scaled
    )


    subcat_features[
        "PC1"
    ] = X_pca[:, 0]


    subcat_features[
        "PC2"
    ] = X_pca[:, 1]


    # --------------------------------------------------------
    # PCA CLUSTER PLOT
    # --------------------------------------------------------

    st.subheader(
        "Demand Cluster Visualization"
    )


    fig, ax = plt.subplots(
        figsize=(12, 8)
    )


    for cluster in sorted(

        subcat_features[
            "Cluster"
        ].unique()

    ):


        cluster_data = (

            subcat_features[

                subcat_features[
                    "Cluster"
                ]

                == cluster

            ]

        )


        ax.scatter(

            cluster_data[
                "PC1"
            ],

            cluster_data[
                "PC2"
            ],

            s=100,

            label=
            cluster_labels[
                cluster
            ]

        )


        # Product labels
        for _, row in (
            cluster_data.iterrows()
        ):


            ax.annotate(

                row[
                    "Sub-Category"
                ],

                (

                    row[
                        "PC1"
                    ],

                    row[
                        "PC2"
                    ]

                ),

                xytext=(5, 5),

                textcoords=
                "offset points",

                fontsize=8

            )


    ax.set_title(

        "Product Demand Segmentation "
        "using K-Means and PCA"

    )


    ax.set_xlabel(

        "Principal Component 1"

    )


    ax.set_ylabel(

        "Principal Component 2"

    )


    ax.legend()

    ax.grid(True)


    st.pyplot(fig)


    # --------------------------------------------------------
    # CLUSTER TABLE
    # --------------------------------------------------------

    st.subheader(
        "Sub-Category Demand Clusters"
    )


    cluster_table = (

        subcat_features[

            [

                "Sub-Category",

                "Demand Cluster",

                "Total_Sales",

                "Sales_Growth_Rate",

                "Sales_Volatility",

                "Average_Order_Value"

            ]

        ]

        .copy()

    )


    st.dataframe(

        cluster_table.round(2),

        use_container_width=True

    )


    # --------------------------------------------------------
    # STOCKING STRATEGIES
    # --------------------------------------------------------

    st.subheader(
        "Recommended Stocking Strategies"
    )


    st.markdown(
        """
**High Volume, Stable Demand:**  
Maintain high inventory levels and use regular replenishment to reduce the risk of stockouts.

**Low Volume, Stable Demand:**  
Maintain lean inventory and use smaller replenishment quantities to reduce holding costs.

**Rapidly Growing Demand:**  
Gradually increase inventory levels and monitor demand frequently to keep pace with growth.

**High-Value, Volatile Demand:**  
Use flexible inventory policies, careful safety-stock planning, and frequent inventory reviews to avoid both stockouts and costly overstocking.
"""
    )

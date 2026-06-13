import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==================================
# LOAD DATA
# ==================================

df = pd.read_csv(
    "data/segmented_customers.csv"
)
centroids = {
    "Standard": [55.3, 49.5],
    "VIP": [86.5, 82.1],
    "Impulsive": [25.7, 79.4],
    "Careful": [88.2, 17.1],
    "Budget": [26.3, 20.9]
}

# ==================================
# SEGMENT NAMES
# ==================================

segment_names = {
    0: "Standard",
    1: "VIP",
    2: "Impulsive",
    3: "Careful",
    4: "Budget"
}

df["Segment"] = df["Cluster"].map(
    segment_names
)

# ==================================
# SIDEBAR
# ==================================

st.sidebar.title("Filters")

selected_segment = st.sidebar.selectbox(
    "Select Segment",
    ["All"] + sorted(df["Segment"].unique())
)

if selected_segment != "All":
    filtered_df = df[
        df["Segment"] == selected_segment
    ]
else:
    filtered_df = df

# ==================================
# TITLE
# ==================================

st.title("Customer Segmentation Dashboard")

st.write(
    """
    Customer segmentation using K-Means clustering.
    Explore customer groups and business insights.
    """
)

st.divider()

# ==================================
# KPI CARDS
# ==================================

total_customers = len(filtered_df)

avg_income = round(
    filtered_df["Annual Income (k$)"].mean(),
    1
)

avg_spending = round(
    filtered_df["Spending Score (1-100)"].mean(),
    1
)

vip_count = len(
    filtered_df[
        filtered_df["Segment"] == "VIP"
    ]
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Customers",
        total_customers
    )

with col2:
    st.metric(
        "VIP Customers",
        vip_count
    )

with col3:
    st.metric(
        "Average Income",
        avg_income
    )

with col4:
    st.metric(
        "Average Spending",
        avg_spending
    )

st.divider()

# ==================================
# SEGMENT DISTRIBUTION
# ==================================

st.subheader(
    "Customer Segment Distribution"
)

segment_counts = (
    df["Segment"]
    .value_counts()
    .reset_index()
)

segment_counts.columns = [
    "Segment",
    "Customers"
]

fig1 = px.bar(
    segment_counts,
    x="Segment",
    y="Customers",
    title="Customers per Segment"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# ==================================
# PIE CHART
# ==================================

st.subheader(
    "Customer Segment Share"
)

fig_pie = px.pie(
    segment_counts,
    names="Segment",
    values="Customers",
    hole=0.4
)

st.plotly_chart(
    fig_pie,
    use_container_width=True
)

# ==================================
# INCOME ANALYSIS
# ==================================

st.subheader(
    "Average Income by Segment"
)

income_df = (
    df.groupby("Segment")
    ["Annual Income (k$)"]
    .mean()
    .reset_index()
)

fig2 = px.bar(
    income_df,
    x="Segment",
    y="Annual Income (k$)",
    title="Average Income by Segment"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ==================================
# SPENDING ANALYSIS
# ==================================

st.subheader(
    "Average Spending by Segment"
)

spending_df = (
    df.groupby("Segment")
    ["Spending Score (1-100)"]
    .mean()
    .reset_index()
)

fig3 = px.bar(
    spending_df,
    x="Segment",
    y="Spending Score (1-100)",
    title="Average Spending Score"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ==================================
# CLUSTER SUMMARY TABLE
# ==================================

st.subheader(
    "Segment Summary"
)

summary = (
    df.groupby("Segment")
    .agg(
        Customers=("CustomerID", "count"),
        Avg_Income=("Annual Income (k$)", "mean"),
        Avg_Spending=("Spending Score (1-100)", "mean")
    )
    .round(1)
)

st.dataframe(
    summary,
    use_container_width=True
)

# ==================================
# CUSTOMER EXPLORER
# ==================================

st.subheader(
    "Customer Explorer"
)

st.dataframe(
    filtered_df,
    use_container_width=True
)

# ==================================
# SEGMENT INFORMATION
# ==================================

segment_info = {
    "VIP":
        "High income and high spending customers. Best target for premium products.",
    "Careful":
        "High income but low spending customers. Good candidates for promotions.",
    "Impulsive":
        "Low income but high spending customers. Responsive to offers and bundles.",
    "Budget":
        "Low income and low spending customers. Focus on discounts.",
    "Standard":
        "Average customers with balanced spending behavior."
}

if selected_segment != "All":
    st.subheader(
        "Segment Insight"
    )

    st.info(
        segment_info[selected_segment]
    )

# ==================================
# BUSINESS INSIGHTS
# ==================================

st.subheader(
    "Business Insights"
)

st.success(
    """
    • VIP customers have the highest spending potential.

    • Careful customers have high income but low spending, making them ideal targets for promotions.

    • Impulsive customers spend heavily despite lower income levels.

    • Budget customers are more responsive to discounts and affordability.

    • Standard customers form the largest customer base.
    """
)

# ==================================
# CUSTOMER SEGMENT PREDICTOR
# ==================================

st.subheader("Predict Customer Segment")

income = st.number_input(
    "Annual Income (k$)",
    min_value=0,
    max_value=200,
    value=50
)

spending = st.number_input(
    "Spending Score (1-100)",
    min_value=1,
    max_value=100,
    value=50
)

if st.button("Predict Segment"):

    min_distance = float("inf")
    predicted_segment = ""

    for segment, center in centroids.items():

        distance = np.sqrt(
            (income - center[0])**2 +
            (spending - center[1])**2
        )

        if distance < min_distance:

            min_distance = distance
            predicted_segment = segment

    st.success(
        f"Predicted Segment: {predicted_segment}"
    )

    if predicted_segment == "VIP":

        st.info(
            "High-income, high-spending customer. Premium offers recommended."
        )

    elif predicted_segment == "Careful":

        st.info(
            "High-income but low-spending customer. Target with promotions."
        )

    elif predicted_segment == "Impulsive":

        st.info(
            "Low-income but high-spending customer. Loyalty programs may work well."
        )

    elif predicted_segment == "Budget":

        st.info(
            "Price-sensitive customer. Discounts and budget products recommended."
        )

    else:

        st.info(
            "Average customer with balanced spending behavior."
        )

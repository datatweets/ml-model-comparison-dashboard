import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Sales Dashboard")

# Generate sample data
@st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=365, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Sales': np.random.randint(100, 1000, 365),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 365),
        'Product': np.random.choice(['A', 'B', 'C'], 365)
    })
    return data

data = load_data()

# Sidebar filters
st.sidebar.header("Filters")
selected_region = st.sidebar.multiselect(
    "Select Region(s):",
    options=data['Region'].unique(),
    default=data['Region'].unique()
)

selected_product = st.sidebar.multiselect(
    "Select Product(s):",
    options=data['Product'].unique(),
    default=data['Product'].unique()
)

# Filter data
filtered_data = data[
    (data['Region'].isin(selected_region)) & 
    (data['Product'].isin(selected_product))
]

# Key metrics
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sales = filtered_data['Sales'].sum()
    st.metric("Total Sales", f"${total_sales:,}")

with col2:
    avg_sales = filtered_data['Sales'].mean()
    st.metric("Average Daily Sales", f"${avg_sales:.0f}")

with col3:
    total_days = len(filtered_data)
    st.metric("Days", total_days)

with col4:
    max_sales = filtered_data['Sales'].max()
    st.metric("Max Daily Sales", f"${max_sales:,}")

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("Sales Over Time")
    fig_line = px.line(filtered_data, x='Date', y='Sales', 
                       title="Daily Sales Trend")
    st.plotly_chart(fig_line, use_container_width=True)

with col2:
    st.subheader("Sales by Region")
    region_sales = filtered_data.groupby('Region')['Sales'].sum().reset_index()
    fig_bar = px.bar(region_sales, x='Region', y='Sales',
                     title="Total Sales by Region")
    st.plotly_chart(fig_bar, use_container_width=True)

# Data table
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.dataframe(filtered_data)
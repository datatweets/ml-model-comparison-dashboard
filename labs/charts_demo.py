import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Data Visualization Demo")

# Generate sample data
np.random.seed(42)
chart_data = pd.DataFrame({
    'Day': range(1, 31),
    'Sales': np.random.randint(50, 200, 30),
    'Customers': np.random.randint(20, 80, 30)
})

# Line chart
st.subheader("Sales Over Time")
st.line_chart(chart_data.set_index('Day')['Sales'])

# Bar chart
st.subheader("Daily Customers")
st.bar_chart(chart_data.set_index('Day')['Customers'])

# Area chart
st.subheader("Sales and Customers")
st.area_chart(chart_data.set_index('Day')[['Sales', 'Customers']])

# Matplotlib integration
st.subheader("Custom Matplotlib Chart")
fig, ax = plt.subplots()
ax.scatter(chart_data['Customers'], chart_data['Sales'])
ax.set_xlabel('Customers')
ax.set_ylabel('Sales')
ax.set_title('Sales vs Customers')
st.pyplot(fig)
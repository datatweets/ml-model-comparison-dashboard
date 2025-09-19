import streamlit as st
import pandas as pd
import numpy as np

st.title("Data Display Demo")

# Create sample data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Score': [85, 92, 78, 95]
}
df = pd.DataFrame(data)

# Display dataframe
st.subheader("Student Data")
st.dataframe(df)

# Display as table
st.subheader("Same Data as Table")
st.table(df)

# Display metrics
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Students", len(df))
with col2:
    st.metric("Average Age", f"{df['Age'].mean():.1f}")
with col3:
    st.metric("Average Score", f"{df['Score'].mean():.1f}")
import streamlit as st
import pandas as pd

st.title("Layout Demo")

# Sidebar
st.sidebar.title("Controls")
chart_type = st.sidebar.selectbox("Choose chart type:", 
                                 ["Line", "Bar", "Area"])

show_data = st.sidebar.checkbox("Show raw data")

# Main content with columns
col1, col2 = st.columns(2)

with col1:
    st.header("Column 1")
    st.write("This is the left column")
    
    # Sample data
    data = pd.DataFrame({
        'x': range(10),
        'y': [i**2 for i in range(10)]
    })
    
    if chart_type == "Line":
        st.line_chart(data.set_index('x'))
    elif chart_type == "Bar":
        st.bar_chart(data.set_index('x'))
    else:
        st.area_chart(data.set_index('x'))

with col2:
    st.header("Column 2")
    st.write("This is the right column")
    
    if show_data:
        st.dataframe(data)
    else:
        st.info("Check the sidebar to show data")
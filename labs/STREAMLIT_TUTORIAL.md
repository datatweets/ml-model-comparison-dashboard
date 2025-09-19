# Streamlit Dashboard Tutorial for Beginners

## Introduction

This tutorial will teach you how to build interactive web applications using Streamlit, starting from the very basics. Streamlit is a Python library that makes it easy to create beautiful web apps for data science and machine learning projects.

## Prerequisites

- Python 3.7 or higher
- Basic Python knowledge
- pip package manager

## Installation

First, install Streamlit:

```bash
pip install streamlit
```

## Tutorial Structure

We'll build progressively from simple text displays to interactive dashboards with data visualization.

---

## Part 1: Your First Streamlit App

### Step 1: Create a Simple Hello World App

Create a file called `hello_world.py`:

```python
import streamlit as st

# Display text
st.title("My First Streamlit App")
st.write("Hello, World!")
st.text("This is simple text")
```

Run it:
```bash
streamlit run hello_world.py
```

### Step 2: Adding Different Text Elements

```python
import streamlit as st

# Different ways to display text
st.title("🌟 My Dashboard")
st.header("This is a header")
st.subheader("This is a subheader")
st.write("This is regular text")
st.markdown("**This is bold text**")
st.markdown("*This is italic text*")
st.code("print('This is code')")
```

---

## Part 2: User Input Widgets

### Step 3: Interactive Widgets

Create `widgets_demo.py`:

```python
import streamlit as st

st.title("Interactive Widgets Demo")

# Text input
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}!")

# Number input
age = st.number_input("Enter your age:", min_value=0, max_value=100, value=25)
st.write(f"You are {age} years old")

# Slider
score = st.slider("Rate your Python skills (1-10):", 1, 10, 5)
st.write(f"Your rating: {score}/10")

# Select box
favorite_color = st.selectbox("Choose your favorite color:", 
                             ["Red", "Blue", "Green", "Yellow"])
st.write(f"Your favorite color is {favorite_color}")

# Checkbox
if st.checkbox("Show secret message"):
    st.write("🎉 You found the secret!")
```

---

## Part 3: Working with Data

### Step 4: Display Data

Create `data_demo.py`:

```python
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
```

---

## Part 4: Data Visualization

### Step 5: Basic Charts

Create `charts_demo.py`:

```python
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
```

---

## Part 5: Layout and Sidebar

### Step 6: Layout Control

Create `layout_demo.py`:

```python
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
```

---

## Part 6: Interactive Dashboard

### Step 7: Complete Dashboard Example

Create `dashboard_demo.py`:

```python
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
```

---

## Part 7: Tips and Best Practices

### Performance Tips

1. **Use `@st.cache_data` for expensive operations:**
```python
@st.cache_data
def load_large_dataset():
    # Expensive data loading
    return pd.read_csv('large_file.csv')
```

2. **Use `st.columns()` for better layout:**
```python
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Metric 1", "100")
```

3. **Add loading spinners for long operations:**
```python
with st.spinner('Loading data...'):
    # Long operation
    time.sleep(2)
    st.success('Data loaded!')
```

### Common Widgets Cheat Sheet

```python
# Text inputs
name = st.text_input("Name")
message = st.text_area("Message")

# Numeric inputs
number = st.number_input("Number", min_value=0, max_value=100)
slider_val = st.slider("Slider", 0, 100, 50)

# Selection
option = st.selectbox("Choose", ["A", "B", "C"])
options = st.multiselect("Choose multiple", ["A", "B", "C"])
radio = st.radio("Pick one", ["X", "Y", "Z"])

# Boolean
checkbox = st.checkbox("Check me")
toggle = st.toggle("Toggle me")

# File upload
file = st.file_uploader("Upload file", type=['csv', 'txt'])

# Date/time
date = st.date_input("Pick a date")
time = st.time_input("Pick a time")
```

## Next Steps

Now that you've learned the basics, you can:

1. **Explore the advanced example** in `app.py` to see how these concepts are used in a real ML dashboard
2. **Add authentication** using `streamlit-authenticator`
3. **Deploy your app** using Streamlit Cloud, Heroku, or other platforms
4. **Learn advanced widgets** like `st.plotly_chart()`, `st.map()`, and custom components

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Streamlit Cheat Sheet](https://cheat-sheet.streamlit.app/)

Happy coding! 🚀
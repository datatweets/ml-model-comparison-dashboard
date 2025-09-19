import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
st.title("Insurance Charges Prediction with Linear Regression")
@st.cache_data
def load_insurance_data():
    data = pd.read_csv('data/insurance.csv')
    data = pd.get_dummies(data, drop_first=True)
    return data
data = load_insurance_data()
st.write("### Dataset Preview")
st.dataframe(data.head(10), use_container_width=True)

X = data.drop('charges', axis=1)
y = data['charges'] 
st.sidebar.title("Model Parameters")
test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5)
random_state = st.sidebar.number_input("Random State", min_value=0, value=42, step=1)
st.write(f"### Model Training with Test Size: {test_size}% and Random State: {random_state}")
if st.sidebar.button("Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    st.subheader("Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test Set Size (%)", f"{test_size}%")
    c2.metric("Train Set Size (%)", f"{100 - test_size}%")
    c3.metric("RMSE", f"{rmse:.2f}")
    c4.metric("R² Score", f"{r2:.2f}")


    st.subheader("Diagnostics")
    cc1, cc2 = st.columns(2)

    with cc1:
        st.markdown("**Actual vs Predicted (Test Set)**")
        df_avp = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.scatter_chart(df_avp, x="Actual", y="Predicted", use_container_width=True)

    with cc2:
        st.markdown("**Residuals (Test Set)**")
        residuals = y_test - y_pred
        hist_counts, hist_bins = np.histogram(residuals, bins=30)
        df_hist = pd.DataFrame({"bin_left": hist_bins[:-1], "count": hist_counts})
        st.bar_chart(df_hist, x="bin_left", y="count", use_container_width=True)

    st.write("### Model Coefficients")
    coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
    st.dataframe(coef_df, use_container_width=True)
else:
    st.write("Adjust the parameters in the sidebar and click 'Train Model' to see results.")
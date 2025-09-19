import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_curve,
    auc,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="ML Model Comparison Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        text-align: center;
        padding: 15px 0;
        border-radius: 5px;
        margin: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🤖 Machine Learning Model Comparison Dashboard")
st.markdown("""
This interactive dashboard allows you to compare different machine learning algorithms on the Breast Cancer dataset.
Select a model, tune its parameters, and analyze its performance in real-time!
""")

# Load and cache data
@st.cache_data
def load_data():
    """Load and return the breast cancer dataset"""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.target_names, data.feature_names

# Load data
df, target_names, feature_names = load_data()

# Sidebar for model selection and parameters
st.sidebar.header("🎛️ Model Configuration")
st.sidebar.markdown("---")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Machine Learning Algorithm",
    ["Logistic Regression", "Decision Tree", "Support Vector Machine (SVM)"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Model Parameters")

# Initialize model variable
model = None

# Model-specific parameters
if model_type == "Logistic Regression":
    st.sidebar.markdown("**Logistic Regression Parameters**")
    
    C = st.sidebar.slider(
        "Regularization Strength (C)",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.01,
        help="Inverse of regularization strength. Smaller values specify stronger regularization"
    )
    
    penalty = st.sidebar.selectbox(
        "Penalty",
        ["l2", "l1", "elasticnet", "none"],
        help="The norm of the penalty"
    )
    
    solver = st.sidebar.selectbox(
        "Solver",
        ["lbfgs", "liblinear", "newton-cg", "saga"],
        help="Algorithm to use in the optimization problem"
    )
    
    max_iter = st.sidebar.slider(
        "Max Iterations",
        min_value=50,
        max_value=1000,
        value=100,
        step=50
    )
    
    # Handle solver-penalty compatibility
    if penalty == "l1" and solver not in ["liblinear", "saga"]:
        solver = "liblinear"
        st.sidebar.warning("Solver changed to 'liblinear' for L1 penalty")
    
    if penalty == "elasticnet":
        solver = "saga"
        l1_ratio = st.sidebar.slider("L1 Ratio", 0.0, 1.0, 0.5)
        model = LogisticRegression(
            C=C, 
            penalty=penalty, 
            solver=solver, 
            max_iter=max_iter,
            l1_ratio=l1_ratio,
            random_state=42
        )
    else:
        model = LogisticRegression(
            C=C, 
            penalty=penalty, 
            solver=solver, 
            max_iter=max_iter,
            random_state=42
        )

elif model_type == "Decision Tree":
    st.sidebar.markdown("**Decision Tree Parameters**")
    
    max_depth = st.sidebar.slider(
        "Max Depth",
        min_value=1,
        max_value=20,
        value=5,
        help="Maximum depth of the tree"
    )
    
    min_samples_split = st.sidebar.slider(
        "Min Samples Split",
        min_value=2,
        max_value=20,
        value=2,
        help="Minimum samples required to split an internal node"
    )
    
    min_samples_leaf = st.sidebar.slider(
        "Min Samples Leaf",
        min_value=1,
        max_value=20,
        value=1,
        help="Minimum samples required to be at a leaf node"
    )
    
    criterion = st.sidebar.selectbox(
        "Criterion",
        ["gini", "entropy"],
        help="Function to measure the quality of a split"
    )
    
    splitter = st.sidebar.selectbox(
        "Splitter",
        ["best", "random"],
        help="Strategy to choose the split at each node"
    )
    
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,
        splitter=splitter,
        random_state=42
    )

else:  # SVM
    st.sidebar.markdown("**SVM Parameters**")
    
    C = st.sidebar.slider(
        "Regularization Parameter (C)",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.01,
        help="Controls the trade-off between smooth decision boundary and classifying training points correctly"
    )
    
    kernel = st.sidebar.selectbox(
        "Kernel",
        ["linear", "poly", "rbf", "sigmoid"],
        help="Kernel type to be used in the algorithm"
    )
    
    gamma = st.sidebar.selectbox(
        "Gamma",
        ["scale", "auto"],
        help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'"
    )
    
    degree = st.sidebar.slider(
        "Degree (for poly kernel)",
        min_value=1,
        max_value=5,
        value=3,
        help="Degree of the polynomial kernel function"
    )
    
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        degree=degree,
        probability=True,  # Enable probability estimates for ROC curve
        random_state=42
    )

# Train-test split configuration
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Training Configuration")

test_size = st.sidebar.slider(
    "Test Set Size",
    min_value=0.1,
    max_value=0.5,
    value=0.3,
    step=0.05,
    help="Proportion of dataset to include in the test split"
)

scale_features = st.sidebar.checkbox(
    "Scale Features",
    value=True,
    help="Apply StandardScaler to features"
)

# Train model button
if st.sidebar.button("🚀 Train Model", type="primary"):
    with st.spinner("Training model..."):
        # Prepare data
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features if selected
        if scale_features:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store results in session state
        st.session_state['trained'] = True
        st.session_state['model'] = model
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test
        st.session_state['y_pred'] = y_pred
        st.session_state['y_pred_proba'] = y_pred_proba
        st.session_state['metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

# Main content area
if 'trained' in st.session_state and st.session_state['trained']:
    # Display metrics
    st.header("📊 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{st.session_state['metrics']['accuracy']:.3f}")
    
    with col2:
        st.metric("Precision", f"{st.session_state['metrics']['precision']:.3f}")
    
    with col3:
        st.metric("Recall", f"{st.session_state['metrics']['recall']:.3f}")
    
    with col4:
        st.metric("F1-Score", f"{st.session_state['metrics']['f1']:.3f}")
    
    st.markdown("---")
    
    # Create two columns for visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Confusion Matrix")
        
        # Create confusion matrix
        cm = confusion_matrix(st.session_state['y_test'], st.session_state['y_pred'])
        
        # Plot confusion matrix using plotly for interactivity
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Benign', 'Predicted Malignant'],
            y=['Actual Benign', 'Actual Malignant'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 20},
            showscale=True
        ))
        
        fig.update_layout(
            title=f"Confusion Matrix - {model_type}",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            height=400,
            width=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display classification report
        with st.expander("📋 Detailed Classification Report"):
            report = classification_report(
                st.session_state['y_test'], 
                st.session_state['y_pred'],
                target_names=['Benign', 'Malignant'],
                output_dict=True
            )
            st.dataframe(pd.DataFrame(report).transpose())
    
    with col2:
        st.subheader("📈 ROC Curve & Optimal Threshold")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(
            st.session_state['y_test'], 
            st.session_state['y_pred_proba']
        )
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        
        # Create ROC curve plot
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, 
            y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], 
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        # Optimal threshold point
        fig.add_trace(go.Scatter(
            x=[optimal_fpr],
            y=[optimal_tpr],
            mode='markers',
            name=f'Optimal Threshold = {optimal_threshold:.3f}',
            marker=dict(color='red', size=12, symbol='star')
        ))
        
        fig.update_layout(
            title=f"ROC Curve - {model_type}",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
            width=500,
            showlegend=True,
            legend=dict(
                yanchor="bottom",
                y=0.02,
                xanchor="right",
                x=0.98
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display optimal threshold metrics
        st.info(f"""
        **Optimal Threshold Analysis:**
        - Threshold Value: {optimal_threshold:.3f}
        - TPR at Optimal: {optimal_tpr:.3f}
        - FPR at Optimal: {optimal_fpr:.3f}
        - AUC Score: {roc_auc:.3f}
        """)
    
    # Feature importance (for Decision Tree)
    if model_type == "Decision Tree":
        st.markdown("---")
        st.subheader("🌳 Feature Importance")
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': st.session_state['model'].feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        fig = go.Figure(data=[
            go.Bar(
                x=feature_importance['Importance'],
                y=feature_importance['Feature'],
                orientation='h',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title="Top 10 Most Important Features",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Model coefficients (for Logistic Regression)
    if model_type == "Logistic Regression":
        st.markdown("---")
        st.subheader("📊 Model Coefficients")
        
        coefficients = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': st.session_state['model'].coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False).head(10)
        
        fig = go.Figure(data=[
            go.Bar(
                x=coefficients['Coefficient'],
                y=coefficients['Feature'],
                orientation='h',
                marker_color=['red' if x < 0 else 'green' for x in coefficients['Coefficient']]
            )
        ])
        
        fig.update_layout(
            title="Top 10 Most Influential Features",
            xaxis_title="Coefficient Value",
            yaxis_title="Feature",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    # Instructions when no model is trained
    st.info("""
    👈 **Get Started:**
    1. Select a machine learning algorithm from the sidebar
    2. Adjust the model parameters as desired
    3. Configure the train-test split ratio
    4. Click the "Train Model" button to see results
    """)
    
    # Display dataset information
    st.header("📊 Dataset Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(df))
    
    with col2:
        st.metric("Total Features", len(feature_names))
    
    with col3:
        st.metric("Target Classes", len(target_names))
    
    # Show data preview
    with st.expander("🔍 Preview Dataset"):
        st.dataframe(df.head(10))
    
    # Show target distribution
    st.subheader("🎯 Target Distribution")
    
    target_counts = df['target'].value_counts()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=['Malignant', 'Benign'],
            values=target_counts.values,
            hole=0.3,
            marker_colors=['#ff6b6b', '#4dabf7']
        )
    ])
    
    fig.update_layout(
        title="Distribution of Diagnoses in Dataset",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with ❤️ using Streamlit | Machine Learning Model Comparison Dashboard</p>
    <p>Dataset: Breast Cancer Wisconsin (Diagnostic)</p>
</div>
""", unsafe_allow_html=True)
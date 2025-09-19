# Machine Learning Model Comparison Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/yourusername/ml-model-comparison-dashboard/main/app.py)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- [Machine Learning Model Comparison Dashboard](#machine-learning-model-comparison-dashboard)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Features](#features)
    - [Core Functionality](#core-functionality)
    - [Supported Algorithms](#supported-algorithms)
    - [Visualization Tools](#visualization-tools)
  - [Demo](#demo)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Step-by-Step Installation](#step-by-step-installation)

## Overview

An interactive Streamlit web application that allows users to compare different machine learning classification algorithms on the Breast Cancer Wisconsin dataset. Users can dynamically adjust model parameters, visualize performance metrics, and identify optimal decision thresholds through ROC curve analysis.

This project serves as both a practical tool for model comparison and an educational resource for understanding machine learning algorithm behavior.

## Features

### Core Functionality
- **Real-time Model Training**: Train models instantly with your selected parameters
- **Interactive Visualizations**: Dynamic plots using Plotly for better interactivity
- **Parameter Tuning**: Adjust model-specific parameters through intuitive UI controls
- **Comprehensive Metrics**: View accuracy, precision, recall, and F1-score at a glance

### Supported Algorithms
1. **Logistic Regression**
   - Regularization strength (C)
   - Penalty types (L1, L2, ElasticNet)
   - Multiple solvers
   - Max iterations control

2. **Decision Tree**
   - Maximum depth control
   - Minimum samples for splitting
   - Criterion selection (Gini, Entropy)
   - Feature importance visualization

3. **Support Vector Machine (SVM)**
   - Kernel selection (Linear, RBF, Polynomial, Sigmoid)
   - Regularization parameter (C)
   - Gamma and degree controls

### Visualization Tools
- **Confusion Matrix**: Interactive heatmap showing prediction results
- **ROC Curve**: With automatic optimal threshold detection using Youden's J statistic
- **Feature Importance**: For Decision Trees
- **Coefficient Analysis**: For Logistic Regression
- **Classification Report**: Detailed per-class metrics

## Demo

[Live Demo Link](https://your-streamlit-app-url.streamlit.app) *(Replace with your deployed app URL)*

## Installation

### Prerequisites
- Python 3.11 or higher
- pip package manager
- Git

### Step-by-Step Installation

1. **Clone the repository**
```bash
git clone https://github.com/datatweets/ml-model-comparison-dashboard.git
cd ml-model-comparison-dashboard
```

2. **Create a virtual environment (recommended)**
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```
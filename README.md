# Computer Vision Model Drift Monitoring Dashboard

## 1. Project Overview

This project is a Streamlit-based web application that serves as a production monitoring dashboard for Computer Vision (CV) models. It is designed to track, evaluate, and visualize different types of model drift in real-time or batch-simulated production environments. The dashboard helps Data Scientists and AI Engineers monitor model performance over time, particularly in industrial manufacturing settings where environmental factors (like lighting and camera positioning) can easily degrade a model's effectiveness.

## 2. Theory: Understanding Model Drift

In machine learning, "drift" refers to the degradation of model performance over time due to changes in the environment, data, or underlying relationships. This project monitors three distinct types of drift:

### A. Data Drift (Covariate Shift)

- **What it is:** Changes in the statistical distribution of the input data (e.g., images becoming darker, blurrier, or taken from a different angle in a manufacturing line).
- **Detection Method:** We extract image embeddings using a pre-trained backbone (like ResNet) or basic image properties (brightness, contrast, sharpness). We then apply statistical tests like **Maximum Mean Discrepancy (MMD)** or the **Kolmogorov-Smirnov (KS) test** on these features to detect statistically significant changes from the original training baseline.

### B. Prediction Drift (Prior Probability Shift)

- **What it is:** Shifts in the model's output predictions over time. Even if ground truth isn't immediately available, seeing a model suddenly output a different distribution of classes can be a strong indicator that something is wrong.
- **Detection Method:** We apply **Kullback-Leibler (KL) Divergence** or **Wasserstein Distance** to compare the probability distributions of predictions on new data against historical baseline predictions.

### C. Concept Drift

- **What it is:** Structural changes in the relationship between input features and target labels. This occurs when the definition of what constitutes a certain class actually changes (requires ground-truth feedback).
- **Detection Method:** We track rolling performance metrics (accuracy, precision, recall, F1-score) over time. We apply the **Page-Hinkley test** or rolling metric degradation thresholds to trigger alerts when model performance objectively drops compared to the training baseline.

## 3. Implementation Details

The application simulates a production environment by allowing users to batch-load folders of images (in Windows Media Photo format or converted equivalents) to simulate a live camera feed.

- **State Management & Authentication:** The app uses Supabase for secure user authentication and metadata storage. Users must log in before accessing the dashboard.
- **Model Support:** The system wraps models using an agnostic `model_wrapper.py` supporting PyTorch, TensorFlow, and ResNet architectures.
- **Micro-Batch Evaluation:** Data streams are evaluated in micro-batches (e.g., every 10-50 images) to provide real-time updates without overwhelming the computation pipeline.
- **Visualizations:** The dashboard uses interactive Plotly charts to display KPI scorecards (Total Drift, Data Drift, Prediction Drift, Concept Drift). Clicking on these metrics reveals deep-dive charts, statistical test outputs (p-values), and automated actionable solutions.

## 4. Tech Stack

- **Frontend / App Framework:** Streamlit
- **Backend & Authentication:** Supabase
- **Storage:** Supabase Storage (for `.pt`, `.h5`, `.onnx` models and baseline datasets)
- **Data Science & ML Libraries:** Python, Pandas, NumPy, Scikit-Learn
- **CV Frameworks:** PyTorch, TensorFlow
- **Drift Detection Mathematics:** `scipy` (for KS-test, MMD, KL Divergence, Wasserstein distance), custom embeddings
- **Visualization:** Plotly, Altair

## 5. File Structure and Usage

```text
cv_drift_dashboard/
├── .streamlit/
│   |── secrets.toml            # Streamlit configurations and Supabase API keys
│   └── config.toml
├── src/
│   ├── auth/                   # Authentication & Session Handlers
│   │   ├── supabase_auth.py    # Login/Signup/Session management via Supabase APIs
│   │   └── session_state.py    # Streamlit local state management wrapper for UI continuity
│   ├── database/               # DB Operations (No UI logic allowed here)
│   │   ├── client.py           # Supabase client instantiation and connection logic
│   │   └── models_db.py        # CRUD operations for model metadata and drift scores
│   ├── utils/                  # Helper Utilities
│   │   ├── image_loader.py     # Windows Media Photo decoding & PIL/NumPy conversion for the CV pipeline
│   │   └── model_wrapper.py    # Universal inference wrapper for standardizing PyTorch/TF/ResNet outputs
│   ├── analytics/              # Math & Statistical Calculations Only
│   │   ├── data_drift.py       # Image embedding extraction and MMD/KS testing logic
│   │   ├── prediction_drift.py # KL Divergence and Wasserstein calculation logic
│   │   └── concept_drift.py    # Performance metric monitoring & Page-Hinkley test algorithms
│   └── ui/                     # UI Views & Layout Compositions
│       ├── components.py       # Reusable Streamlit components (KPI metrics cards, custom dividers)
│       ├── views_auth.py       # Authentication and onboarding view templates
│       ├── views_upload.py     # Forms and file upload UI for models and baseline datasets
│       └── views_dashboard.py  # Plotly/Altair visualizations and drill-down insights logic
├── main.py                     # Main application entry point orchestrating views & state
├── supabase_schema.sql         # SQL schema definitions for the Supabase Postgres database
└── requirements.txt            # Explicit production Python dependencies
```

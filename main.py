# Responsible and Safe AI - External Practical Set C
# Enrollment No: 23CE055
# Source code exported from the executed Google Colab notebook.

# ========================================================================
# NOTEBOOK CELL 1
# ========================================================================

# STEP 1: Install required libraries

!pip install -q fairlearn

print("Libraries installed successfully!")


# ========================================================================
# NOTEBOOK CELL 2
# ========================================================================

# STEP 2: Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference
)

from fairlearn.postprocessing import ThresholdOptimizer

print("All imports completed successfully!")


# ========================================================================
# NOTEBOOK CELL 3
# ========================================================================

# STEP 3: Upload Adult Census Income Dataset

from google.colab import files

uploaded = files.upload()

print("Dataset uploaded successfully!")


# ========================================================================
# NOTEBOOK CELL 4
# ========================================================================

# STEP 4: Extract and Load the ZIP Dataset

import zipfile
import io
import pandas as pd

# Get uploaded ZIP filename
filename = list(uploaded.keys())[0]

print("Uploaded file:", filename)

# Extract ZIP and identify CSV file
with zipfile.ZipFile(io.BytesIO(uploaded[filename]), 'r') as zip_ref:

    csv_files = [
        f for f in zip_ref.namelist()
        if f.lower().endswith('.csv') and not f.startswith('__MACOSX/')
    ]

    print("CSV files found:", csv_files)

    if not csv_files:
        raise FileNotFoundError("No CSV file found inside ZIP!")

    # Read the first CSV file
    with zip_ref.open(csv_files[0]) as file:
        df = pd.read_csv(file, skipinitialspace=True)

print("\nDataset loaded successfully!")

print("\nDataset Shape:", df.shape)

print("\nFirst 5 Records:")
display(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())


# ========================================================================
# NOTEBOOK CELL 5
# ========================================================================

# STEP 6: Data Cleaning

# Remove leading and trailing spaces from column names
df.columns = df.columns.str.strip()

# Remove whitespace from string values
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.strip()

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Remove rows containing missing values
df.dropna(inplace=True)

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Normalize income labels
df['income'] = df['income'].str.replace('.', '', regex=False).str.strip()

# Convert target variable into binary format
df['income'] = df['income'].map({
    '<=50K': 0,
    '>50K': 1
})

# Remove any rows with invalid target labels
df.dropna(subset=['income'], inplace=True)
df['income'] = df['income'].astype(int)

print("Data cleaning completed!")
print("\nCleaned Dataset Shape:", df.shape)

print("\nIncome Distribution:")
print(df['income'].value_counts())

print("\nSex Distribution:")
print(df['sex'].value_counts())

print("\nRemaining Missing Values:", df.isnull().sum().sum())


# ========================================================================
# NOTEBOOK CELL 6
# ========================================================================

# STEP 7: Separate Features and Target

# Sensitive attribute (used ONLY for fairness analysis)
sensitive_features = df['sex'].copy()

# Target variable
y = df['income'].copy()

# Model features (exclude income and sex)
X = df.drop(columns=['income', 'sex'])

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)
print("Sensitive Attribute Shape:", sensitive_features.shape)

print("\nSensitive Attribute:")
print(sensitive_features.value_counts())

print("\nTarget Classes:")
print(y.value_counts())


# ========================================================================
# NOTEBOOK CELL 7
# ========================================================================

# STEP 8: Identify Feature Types

numeric_features = X.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

categorical_features = X.select_dtypes(
    include=['object', 'category']
).columns.tolist()

print("Numerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ========================================================================
# NOTEBOOK CELL 8
# ========================================================================

# STEP 9: Preprocessing and Train-Test Split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Split data
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X,
    y,
    sensitive_features,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Data splitting completed!")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining income distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting income distribution:")
print(y_test.value_counts(normalize=True))


# ========================================================================
# NOTEBOOK CELL 9
# ========================================================================

# STEP 10: Train Logistic Regression Model

from sklearn.linear_model import LogisticRegression

# Create complete machine learning pipeline
baseline_model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train model
baseline_model.fit(X_train, y_train)

# Make predictions
y_pred_baseline = baseline_model.predict(X_test)

# Calculate accuracy
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)

print("Baseline Logistic Regression Model")
print("-----------------------------------")
print("Accuracy:", round(baseline_accuracy, 4))
print("Model trained successfully!")


# ========================================================================
# NOTEBOOK CELL 10
# ========================================================================

# STEP 11: Baseline Fairness Audit

metrics = {
    'Accuracy': accuracy_score,
    'Selection Rate': selection_rate,
    'TPR': true_positive_rate,
    'FPR': false_positive_rate
}

baseline_metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred_baseline,
    sensitive_features=A_test
)

print("BASELINE FAIRNESS METRICS")
print("=========================")

display(baseline_metric_frame.by_group)

print("\nOverall Metrics:")
display(baseline_metric_frame.overall)


# ========================================================================
# NOTEBOOK CELL 11
# ========================================================================

# STEP 12: Calculate Fairness Metrics

baseline_dp_diff = demographic_parity_difference(
    y_test,
    y_pred_baseline,
    sensitive_features=A_test
)

baseline_dp_ratio = demographic_parity_ratio(
    y_test,
    y_pred_baseline,
    sensitive_features=A_test
)

baseline_eo_diff = equalized_odds_difference(
    y_test,
    y_pred_baseline,
    sensitive_features=A_test
)

print("BASELINE FAIRNESS RESULTS")
print("=========================")

print("Overall Accuracy:", round(baseline_accuracy, 4))

print("Demographic Parity Difference:",
      round(baseline_dp_diff, 4))

print("Demographic Parity Ratio:",
      round(baseline_dp_ratio, 4))

print("Equalized Odds Difference:",
      round(baseline_eo_diff, 4))

print("\nFour-Fifths Rule:",
      "PASS" if baseline_dp_ratio >= 0.80 else "FAIL - Potential Bias")


# ========================================================================
# NOTEBOOK CELL 12
# ========================================================================

# STEP 13: ThresholdOptimizer - Demographic Parity

from fairlearn.postprocessing import ThresholdOptimizer

dp_optimizer = ThresholdOptimizer(
    estimator=baseline_model,
    constraints='demographic_parity',
    objective='accuracy_score',
    predict_method='predict_proba',
    prefit=True
)

# Fit fairness postprocessor
dp_optimizer.fit(
    X_train,
    y_train,
    sensitive_features=A_train
)

# Generate mitigated predictions
y_pred_dp = dp_optimizer.predict(
    X_test,
    sensitive_features=A_test,
    random_state=42
)

dp_accuracy = accuracy_score(y_test, y_pred_dp)

print("DEMOGRAPHIC PARITY MITIGATION")
print("============================")
print("Accuracy:", round(dp_accuracy, 4))
print("Mitigation completed successfully!")


# ========================================================================
# NOTEBOOK CELL 13
# ========================================================================

# STEP 14: Evaluate Demographic Parity Model

dp_metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred_dp,
    sensitive_features=A_test
)

dp_diff = demographic_parity_difference(
    y_test, y_pred_dp, sensitive_features=A_test
)

dp_ratio = demographic_parity_ratio(
    y_test, y_pred_dp, sensitive_features=A_test
)

dp_eo_diff = equalized_odds_difference(
    y_test, y_pred_dp, sensitive_features=A_test
)

print("DEMOGRAPHIC PARITY FAIRNESS METRICS")
print("===================================")

display(dp_metric_frame.by_group)

print("\nDemographic Parity Difference:", round(dp_diff, 4))
print("Demographic Parity Ratio:", round(dp_ratio, 4))
print("Equalized Odds Difference:", round(dp_eo_diff, 4))


# ========================================================================
# NOTEBOOK CELL 14
# ========================================================================

# STEP 15: ThresholdOptimizer - Equalized Odds

eo_optimizer = ThresholdOptimizer(
    estimator=baseline_model,
    constraints='equalized_odds',
    objective='accuracy_score',
    predict_method='predict_proba',
    prefit=True
)

# Fit fairness postprocessor
eo_optimizer.fit(
    X_train,
    y_train,
    sensitive_features=A_train
)

# Generate mitigated predictions
y_pred_eo = eo_optimizer.predict(
    X_test,
    sensitive_features=A_test,
    random_state=42
)

eo_accuracy = accuracy_score(y_test, y_pred_eo)

print("EQUALIZED ODDS MITIGATION")
print("=========================")
print("Accuracy:", round(eo_accuracy, 4))
print("Mitigation completed successfully!")


# ========================================================================
# NOTEBOOK CELL 15
# ========================================================================

# STEP 16: Evaluate Equalized Odds Model

eo_metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred_eo,
    sensitive_features=A_test
)

eo_dp_diff = demographic_parity_difference(
    y_test, y_pred_eo, sensitive_features=A_test
)

eo_dp_ratio = demographic_parity_ratio(
    y_test, y_pred_eo, sensitive_features=A_test
)

eo_diff = equalized_odds_difference(
    y_test, y_pred_eo, sensitive_features=A_test
)

print("EQUALIZED ODDS FAIRNESS METRICS")
print("===============================")

display(eo_metric_frame.by_group)

print("\nDemographic Parity Difference:", round(eo_dp_diff, 4))
print("Demographic Parity Ratio:", round(eo_dp_ratio, 4))
print("Equalized Odds Difference:", round(eo_diff, 4))


# ========================================================================
# NOTEBOOK CELL 16
# ========================================================================

# STEP 17: Model Comparison Table

def get_selection_rates(metric_frame):
    group_rates = metric_frame.by_group['Selection Rate']
    return group_rates['Male'], group_rates['Female']


base_m, base_f = get_selection_rates(baseline_metric_frame)
dp_m, dp_f = get_selection_rates(dp_metric_frame)
eo_m, eo_f = get_selection_rates(eo_metric_frame)

comparison = pd.DataFrame({
    'Model': [
        'Baseline',
        'TO: Demographic Parity',
        'TO: Equalized Odds'
    ],
    'Accuracy': [
        baseline_accuracy,
        dp_accuracy,
        eo_accuracy
    ],
    'Selection Rate Male': [
        base_m, dp_m, eo_m
    ],
    'Selection Rate Female': [
        base_f, dp_f, eo_f
    ],
    'DP Difference': [
        baseline_dp_diff, dp_diff, eo_dp_diff
    ],
    'DP Ratio': [
        baseline_dp_ratio, dp_ratio, eo_dp_ratio
    ],
    'EO Difference': [
        baseline_eo_diff, dp_eo_diff, eo_diff
    ]
})

print("FINAL MODEL COMPARISON")
print("======================")

display(comparison.round(4))


# ========================================================================
# NOTEBOOK CELL 17
# ========================================================================

# STEP 18: Calculate Accuracy Loss

accuracy_loss_dp = baseline_accuracy - dp_accuracy
accuracy_loss_eo = baseline_accuracy - eo_accuracy

print("ACCURACY VS FAIRNESS ANALYSIS")
print("=============================")

print("Baseline Accuracy:", round(baseline_accuracy, 4))

print("\nDemographic Parity:")
print("Accuracy:", round(dp_accuracy, 4))
print("Accuracy Loss:", round(accuracy_loss_dp, 4))

print("\nEqualized Odds:")
print("Accuracy:", round(eo_accuracy, 4))
print("Accuracy Loss:", round(accuracy_loss_eo, 4))

if accuracy_loss_dp >= 0:
    print("\nDP Accuracy Loss (%):",
          round(accuracy_loss_dp * 100, 2))
else:
    print("\nDP Accuracy Improvement (%):",
          round(abs(accuracy_loss_dp) * 100, 2))

if accuracy_loss_eo >= 0:
    print("EO Accuracy Loss (%):",
          round(accuracy_loss_eo * 100, 2))
else:
    print("EO Accuracy Improvement (%):",
          round(abs(accuracy_loss_eo) * 100, 2))


# ========================================================================
# NOTEBOOK CELL 18
# ========================================================================

# STEP 19: Selection Rate Comparison Graph

models = ['Baseline', 'Demographic Parity', 'Equalized Odds']

male_rates = [base_m, dp_m, eo_m]
female_rates = [base_f, dp_f, eo_f]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(x - width/2, male_rates, width, label='Male')
plt.bar(x + width/2, female_rates, width, label='Female')

plt.xlabel('Models')
plt.ylabel('Selection Rate')
plt.title('Selection Rate by Sex - Fairness Comparison')

plt.xticks(x, models, rotation=10)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig('selection_rate_by_sex.png', dpi=300)

plt.show()

print("Graph saved as selection_rate_by_sex.png")


# ========================================================================
# NOTEBOOK CELL 19
# ========================================================================

# STEP 20: Save Final Results

comparison.to_csv('fairness_comparison.csv', index=False)

print("Comparison table saved successfully!")
print("Files generated:")
print("1. selection_rate_by_sex.png")
print("2. fairness_comparison.csv")


# ========================================================================
# NOTEBOOK CELL 20
# ========================================================================

# STEP 21: Technical Questions and Answers

print("=" * 65)
print("TECHNICAL QUESTIONS - RESPONSIBLE AND SAFE AI")
print("=" * 65)

# Q1
print("\nQ1. BASELINE MODEL PERFORMANCE")
print("--------------------------------")
print("Overall Accuracy:", round(baseline_accuracy, 4))

print("\nGroup-wise metrics:")
display(baseline_metric_frame.by_group)

print("Disadvantaged group:",
      "Female" if base_f < base_m else "Male"
      if base_m < base_f else "Neither based on selection rate")

# Q2
print("\nQ2. BASELINE FAIRNESS AUDIT")
print("--------------------------------")
print("Demographic Parity Difference:",
      round(baseline_dp_diff, 4))
print("Demographic Parity Ratio:",
      round(baseline_dp_ratio, 4))
print("Equalized Odds Difference:",
      round(baseline_eo_diff, 4))

print("Four-Fifths Rule:",
      "PASS" if baseline_dp_ratio >= 0.80 else "FAIL")

print("The model can exhibit bias even without using sex",
      "because other features may be correlated with sex.")

# Q3
print("\nQ3. THRESHOLD OPTIMIZER COMPARISON")
print("--------------------------------")
print("Demographic Parity:")
print("DP Difference:", round(dp_diff, 4))
print("EO Difference:", round(dp_eo_diff, 4))

print("\nEqualized Odds:")
print("DP Difference:", round(eo_dp_diff, 4))
print("EO Difference:", round(eo_diff, 4))

# Q4
print("\nQ4. ACCURACY-FAIRNESS TRADE-OFF")
print("--------------------------------")
print("Baseline Accuracy:", round(baseline_accuracy, 4))
print("DP Accuracy:", round(dp_accuracy, 4))
print("EO Accuracy:", round(eo_accuracy, 4))

print("\nAccuracy change with DP:",
      round((dp_accuracy - baseline_accuracy) * 100, 2), "percentage points")

print("Accuracy change with EO:",
      round((eo_accuracy - baseline_accuracy) * 100, 2), "percentage points")

if dp_diff < eo_dp_diff:
    print("\nDP achieves better demographic parity.")
else:
    print("\nEO achieves better or equal demographic parity.")

if eo_diff < dp_eo_diff:
    print("EO achieves better equalized odds.")
else:
    print("DP achieves better or equal equalized odds.")

print("\nLimitation: Fairness mitigation may reduce predictive",
      "accuracy and requires sensitive attribute information.")


# ========================================================================
# NOTEBOOK CELL 21
# ========================================================================

# STEP 22: Download Generated Files

from google.colab import files

files.download('selection_rate_by_sex.png')
files.download('fairness_comparison.csv')

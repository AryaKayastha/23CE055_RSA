# Fairness Audit of Logistic Regression Using Fairlearn

**Course:** Responsible and Safe AI (AIML4001)  
**Practical Examination:** External Practical Examination – Set C  
**Student Name:** Arya Kayastha  
**Enrollment Number:** 23CE055

## Project Overview

This project performs a fairness audit of a Logistic Regression model using the Adult Census Income dataset. It evaluates whether model predictions differ across Male and Female groups and applies Fairlearn's `ThresholdOptimizer` to explore fairness mitigation.

The project compares the baseline model with two fairness constraints:

- Demographic Parity
- Equalized Odds

The objective is to understand the trade-offs between predictive performance and group fairness.

## Objectives

1. Load and clean the Adult Census Income dataset.
2. Preprocess numerical and categorical features.
3. Train a Logistic Regression classifier without using the `sex` feature as an input.
4. Evaluate accuracy, selection rate, true positive rate (TPR), and false positive rate (FPR) by sex.
5. Calculate demographic parity difference, demographic parity ratio, and equalized odds difference.
6. Apply Fairlearn's `ThresholdOptimizer` with demographic parity and equalized odds constraints.
7. Compare model performance and fairness before and after mitigation.

## Dataset

- **Dataset:** Adult Census Income (`adult.csv`)
- **Original size:** 32,561 rows and 15 columns
- **Rows retained after cleaning:** 30,139
- **Target column:** `income`
- **Sensitive feature:** `sex`

Rows containing `?` values were removed during cleaning. The `sex` column was retained for group-based fairness evaluation but excluded from the model's predictive features.

## Methodology

### 1. Data preprocessing

- Removed rows containing `?` values.
- Removed missing values and duplicate rows.
- Separated the target (`income`) from the input features.
- Excluded `sex` from the model input features.
- Applied one-hot encoding to categorical features.
- Scaled numerical features.

### 2. Model training

- **Algorithm:** Logistic Regression
- **Maximum iterations:** 1,000
- **Train-test split:** 70:30
- **Stratification:** Yes
- **Random state:** 42

### 3. Fairness evaluation

Fairlearn's `MetricFrame` was used to calculate:

- Accuracy
- Selection rate
- True positive rate (TPR)
- False positive rate (FPR)

The following fairness measures were also calculated:

- Demographic parity difference
- Demographic parity ratio (disparate impact)
- Equalized odds difference

### 4. Fairness mitigation

Fairlearn's `ThresholdOptimizer` was applied using:

- `demographic_parity`
- `equalized_odds`

The mitigated models were compared with the baseline model in terms of accuracy and fairness metrics.

## Key Results

### Baseline model

| Metric | Result |
|---|---:|
| Test accuracy | 0.8524 |
| Demographic parity difference | 0.1759 |
| Demographic parity ratio | 0.3212 |
| Equalized odds difference | 0.0906 |

### Group metrics – baseline model

| Group | Accuracy | Selection Rate | TPR | FPR |
|---|---:|---:|---:|---:|
| Female | 0.9239 | 0.0832 | 0.5319 | 0.0265 |
| Male | 0.8180 | 0.2591 | 0.6225 | 0.0922 |

### Mitigation comparison

| Model | Accuracy | Male Selection Rate | Female Selection Rate | DP Difference | DP Ratio | EO Difference |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.8524 | 0.2591 | 0.0832 | 0.1759 | 0.3212 | 0.0906 |
| Demographic Parity | 0.8322 | 0.1732 | 0.1620 | 0.0112 | 0.9356 | 0.3109 |
| Equalized Odds | 0.8371 | 0.2074 | 0.1037 | 0.1037 | 0.5000 | 0.0379 |

**Note:** Results are from the executed notebook. Small differences may occur if the environment, library versions, or execution settings differ.

## Observations

- The baseline model has a noticeable difference in selection rates between Male and Female groups.
- Demographic parity mitigation substantially reduces the selection-rate difference, with a small reduction in accuracy.
- Equalized odds mitigation reduces the equalized-odds difference, while demographic parity remains less balanced than in the demographic parity model.
- The results illustrate that improving one fairness criterion may affect another fairness criterion and model accuracy.

## Repository Structure

```text
23CE055_RSA/
│
├── README.md
├── RSA_External_Practical_Set_C.ipynb
├── main.py
├── selection_rate_by_sex.png
├── fairness_comparison.csv
└── 23CE055_External_Practical_Exam.pdf
```

You may also include the editable Word report in the repository.

## How to Run

### Google Colab

1. Open `RSA_External_Practical_Set_C.ipynb` in Google Colab.
2. Upload `adult.csv` when prompted, or update the dataset path in the notebook.
3. Run the notebook cells from top to bottom.
4. Install any required dependencies if prompted.

### Local environment

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn fairlearn matplotlib
```

Then run:

```bash
python main.py
```

Ensure that the Adult Census Income dataset is available at the path expected by the script.

## Tools and Libraries

- Python
- Google Colab / Jupyter Notebook
- pandas
- NumPy
- scikit-learn
- Fairlearn
- Matplotlib

## Conclusion

This practical demonstrates how to audit a machine-learning classifier for group fairness and how to use Fairlearn's `ThresholdOptimizer` to mitigate selected fairness disparities.

The comparison highlights trade-offs between accuracy, demographic parity, and equalized odds. Fairness assessment should consider the application context and consequences of different types of errors rather than relying on a single metric.

## Author

**Arya Kayastha**  
**Enrollment Number:** 23CE055  
**GitHub Repository:** https://github.com/AryaKayastha/23CE055_RSA

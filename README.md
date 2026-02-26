# Customer Churn Prediction - End-to-End ML Project

## 📌 Problem Statement
Predict customer churn for a telecom company to help reduce customer loss and improve retention strategies.

## 🧠 Business Impact
Reducing churn by 5% can increase company profits by 25-95%.

## 📊 Dataset
IBM Telco Customer Churn Dataset

## ⚙️ Tech Stack
- Python
- Scikit-learn
- XGBoost
- SHAP
- Pandas, NumPy, Seaborn

## 🚀 ML Pipeline
1. Data Cleaning & EDA
2. Missing Value Handling
3. Feature Engineering
4. Pipeline Preprocessing
5. Model Comparison (5 Models)
6. Hyperparameter Tuning
7. Cross-Validation
8. Model Interpretation (SHAP)

## 📈 Model Performance
| Model | F1 Score | ROC-AUC |
|-------|----------|---------|
| Logistic Regression (best on test) | 0.60 | 0.84 |

## 🔍 Key Insights
- Month-to-month contracts have highest churn
- High monthly charges increase churn risk
- Low tenure customers churn more

## 📦 Project Structure
Modular production-level code with reusable pipelines.

```text
.
├── data
│   ├── raw
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── processed
│       ├── train.csv
│       └── test.csv
├── models
│   └── best_churn_model.pkl
├── reports
│   └── classification_report.txt
├── notebook
│   └── 01_eda.ipynb
├── src
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── evaluate.py
├── main.py
└── requirements.txt
```

## 🔧 How to Run (GitHub Ready)

1. **Clone the repository**
   - `git clone <your-repo-url>`
   - `cd customer-churn-ml`
2. **Create and activate a virtual environment (recommended)**
   - `python -m venv .venv`
   - Windows: `.\.venv\Scripts\activate`
3. **Install dependencies**
   - `pip install -r requirements.txt`
4. **Run the end-to-end pipeline**
   - `python main.py`

This will:
- Load the raw IBM Telco churn data.
- Create cleaned, model-ready splits under `data/processed/`.
- Train, compare, and tune multiple models.
- Evaluate the best model and write a text report to `reports/classification_report.txt`.
- Save the final model artifact to `models/best_churn_model.pkl`.

## 📁 Outputs for Stakeholders

- **Processed datasets**: `data/processed/train.csv`, `data/processed/test.csv` (ready for BI tools or further analysis).
- **Model evaluation report**: `reports/classification_report.txt` (includes F1, ROC-AUC, and full classification report).
- **Production model**: `models/best_churn_model.pkl` (can be loaded in an API or batch scoring job).
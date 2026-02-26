from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_recall_curve
import os
import matplotlib.pyplot as plt
import shap

def evaluate_model(pipeline, X_test, y_test, report_dir: str | None = None):
    """Evaluate model performance using multiple metrics and optionally save a report."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("F1 Score:", f1)
    print("ROC-AUC:", roc_auc)
    print("\nClassification Report:")
    report_text = classification_report(y_test, y_pred)
    print(report_text)

    if report_dir is not None:
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Customer Churn Model Evaluation\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report_text)


def explain_with_shap(pipeline, X_test, y_test):
    """Generate SHAP explanations for model predictions"""
    # Transform X_test through preprocessor
    X_test_transformed = pipeline.named_steps['preprocessor'].transform(X_test)
    
    # Extract the trained model from pipeline
    model = pipeline.named_steps['model']
    
    # Create SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test_transformed)
    
    # Plot SHAP summary
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values, X_test_transformed)
    plt.tight_layout()
    plt.show()
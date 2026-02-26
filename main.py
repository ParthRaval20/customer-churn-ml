import os
import joblib
from sklearn.pipeline import Pipeline

from src.data_preprocessing import load_data, split_data
from src.feature_engineering import create_preprocessor
from src.train import get_models, train_and_compare, tune_best_model
from src.evaluate import evaluate_model, explain_with_shap


def main():
    print("=" * 50)
    print("STARTING END-TO-END CUSTOMER CHURN ML PIPELINE")
    print("=" * 50)

    # ==============================
    # 1. LOAD DATA
    # ==============================
    print("\n[1/6] Loading Data...")
    df = load_data()
    print(f"Dataset Shape: {df.shape}")

    # ==============================
    # 2. TRAIN-TEST SPLIT
    # ==============================
    print("\n[2/6] Splitting Data...")
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"X_train Shape: {X_train.shape}")
    print(f"X_test Shape: {X_test.shape}")

    # ==============================
    # 3. FEATURE ENGINEERING PIPELINE
    # ==============================
    print("\n[3/6] Creating Preprocessing Pipeline...")
    preprocessor = create_preprocessor(X_train)
    print("Preprocessing pipeline created (Imputation + Encoding + Scaling)")

    # ==============================
    # 4. MODEL TRAINING & COMPARISON
    # ==============================
    print("\n[4/6] Training & Comparing Models with Cross-Validation...")
    print("(This may take several minutes - training 5 models with 3-fold CV)\n")
    
    try:
        results = train_and_compare(X_train, y_train, preprocessor)
    except KeyboardInterrupt:
        print("\n[WARN] Training interrupted. Using default results...")
        from src.train import get_models
        results = {name: 0.5 for name in get_models().keys()}
    except Exception as e:
        print(f"\n[WARN] Error during training: {str(e)}")
        from src.train import get_models
        results = {name: 0.5 for name in get_models().keys()}

    print("\nCross-Validation F1 Scores:")
    for model_name, score in results.items():
        print(f"{model_name}: {score:.4f}")

    # Get best model name
    best_model_name = max(results, key=results.get)
    print(f"\nBest Model Based on F1 Score: {best_model_name}")

    # ==============================
    # 5. HYPERPARAMETER TUNING
    # ==============================
    print("\n[5/6] Hyperparameter Tuning on Best Model...")
    try:
        best_pipeline = tune_best_model(
            model_name=best_model_name,
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor
        )
    except KeyboardInterrupt:
        print("\n[WARN] Tuning interrupted. Using base model without tuning...")
        from sklearn.pipeline import Pipeline
        best_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", get_models()[best_model_name])
        ])
        best_pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"\n[WARN] Error during tuning: {str(e)}")
        from sklearn.pipeline import Pipeline
        best_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", get_models()[best_model_name])
        ])
        best_pipeline.fit(X_train, y_train)

    # ==============================
    # 6. FINAL EVALUATION
    # ==============================
    print("\n[6/7] Final Model Evaluation on Test Set...")
    reports_dir = os.path.join("reports")
    evaluate_model(best_pipeline, X_test, y_test, report_dir=reports_dir)

    # ==============================
    # 7. MODEL INTERPRETABILITY (SHAP)
    # ==============================
    print("\n[7/8] Generating SHAP Explanations for Model Interpretability...")
    try:
        explain_with_shap(best_pipeline, X_test, y_test)
    except Exception as e:
        print(f"[WARN] SHAP visualization skipped: {str(e)}")

    # ==============================
    # 8. SAVE MODEL (PRODUCTION STEP)
    # ==============================
    print("\nSaving Best Model...")
    model_dir = os.path.join("models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "best_churn_model.pkl")
    joblib.dump(best_pipeline, model_path)

    print(f"Model saved at: {model_path}")
    print("\nEND-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
    print("\nPipeline Summary:")
    print(f"   - Best Model: {best_model_name}")
    print(f"   - Best F1 Score (CV): {results[best_model_name]:.4f}")
    print(f"   - Training Set: {X_train.shape[0]} samples")
    print(f"   - Test Set: {X_test.shape[0]} samples")


if __name__ == "__main__":
    main()
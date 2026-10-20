from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from preprocessing import (
    create_preprocessor,
    prepare_features_and_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_data():
    """Load the Heart Disease dataset."""

    df = pd.read_csv(DATA_PATH)

    X, y = prepare_features_and_target(df)

    return X, y


def build_logistic_regression_pipeline():
    """Build Logistic Regression pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_random_forest_pipeline():
    """Build Random Forest pipeline."""

    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def evaluate_model(model, X_test, y_test):
    """Evaluate a fitted binary classification model."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    return metrics


def main():
    X, y = load_data()

    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # ---------------------------------------------------------
    # 1. Logistic Regression
    # ---------------------------------------------------------

    logistic_model = build_logistic_regression_pipeline()

    logistic_model.fit(X_train, y_train)

    logistic_metrics = evaluate_model(
        logistic_model,
        X_test,
        y_test,
    )

    print("\nLogistic Regression")
    print("-" * 30)

    for metric, value in logistic_metrics.items():
        print(f"{metric}: {value:.4f}")

    # ---------------------------------------------------------
    # 2. Random Forest
    # ---------------------------------------------------------

    random_forest_model = build_random_forest_pipeline()

    random_forest_model.fit(X_train, y_train)

    random_forest_metrics = evaluate_model(
        random_forest_model,
        X_test,
        y_test,
    )

    print("\nRandom Forest")
    print("-" * 30)

    for metric, value in random_forest_metrics.items():
        print(f"{metric}: {value:.4f}")

    # ---------------------------------------------------------
    # 3. Random Forest Hyperparameter Tuning
    # ---------------------------------------------------------

    print("\nStarting Random Forest hyperparameter tuning...")

    tuning_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    parameter_grid = {
        "classifier__n_estimators": [
            100,
            200,
            300,
        ],
        "classifier__max_depth": [
            None,
            5,
            10,
        ],
        "classifier__min_samples_split": [
            2,
            5,
        ],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid_search = GridSearchCV(
        estimator=tuning_pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        f"\nBest cross-validation ROC-AUC: "
        f"{grid_search.best_score_:.4f}"
    )

    # ---------------------------------------------------------
    # 4. Evaluate tuned model
    # ---------------------------------------------------------

    tuned_model = grid_search.best_estimator_

    tuned_metrics = evaluate_model(
        tuned_model,
        X_test,
        y_test,
    )

    print("\nTuned Random Forest")
    print("-" * 30)

    for metric, value in tuned_metrics.items():
        print(f"{metric}: {value:.4f}")

    # ---------------------------------------------------------
    # 5. Compare models
    # ---------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                **logistic_metrics,
            },
            {
                "model": "Random Forest",
                **random_forest_metrics,
            },
            {
                "model": "Tuned Random Forest",
                **tuned_metrics,
            },
        ]
    )

    print("\nModel Comparison")
    print("=" * 70)
    print(comparison.to_string(index=False))
    results_dir = PROJECT_ROOT / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    comparison.to_csv(
          results_dir / "model_comparison.csv",
          index=False,
    )

    # ---------------------------------------------------------
    # 6. Select final model
    # ---------------------------------------------------------

    final_model = tuned_model

    model_path = MODEL_DIR / "heart_disease_model.joblib"

    joblib.dump(final_model, model_path)

    print("\nFinal model saved to:")
    print(model_path)


if __name__ == "__main__":
    main()

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
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


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"

MODEL_DIR = PROJECT_ROOT / "models"

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

RESULTS_DIR = PROJECT_ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# ============================================================
# DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    X, y = prepare_features_and_target(df)

    return X, y


# ============================================================
# MODELS
# ============================================================

def build_logistic_regression():

    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_random_forest():

    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
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


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
        ),
        "recall": recall_score(
            y_test,
            predictions,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    return metrics, predictions, probabilities


# ============================================================
# MLflow LOGGING
# ============================================================

def log_model_run(
    run_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    parameters,
):

    with mlflow.start_run(
        run_name=run_name
    ):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        metrics, predictions, probabilities = (
            evaluate_model(
                model,
                X_test,
                y_test,
            )
        )

        # ----------------------------------------------------
        # Log parameters
        # ----------------------------------------------------

        for parameter_name, parameter_value in parameters.items():

            mlflow.log_param(
                parameter_name,
                parameter_value,
            )

        mlflow.log_param(
            "test_size",
            0.2,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        # ----------------------------------------------------
        # Log metrics
        # ----------------------------------------------------

        for metric_name, metric_value in metrics.items():

            mlflow.log_metric(
                metric_name,
                metric_value,
            )

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        cm = confusion_matrix(
            y_test,
            predictions,
        )

        fig, ax = plt.subplots(
            figsize=(6, 6)
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=[
                "No Disease",
                "Disease",
            ],
        )

        display.plot(ax=ax)

        ax.set_title(
            f"{run_name} - Confusion Matrix"
        )

        safe_name = (
            run_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        confusion_path = (
            FIGURE_DIR /
            f"{safe_name}_confusion_matrix.png"
        )

        fig.savefig(
            confusion_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        mlflow.log_artifact(
            str(confusion_path),
            artifact_path="figures",
        )

        # ----------------------------------------------------
        # ROC curve
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )

        RocCurveDisplay.from_predictions(
            y_test,
            probabilities,
            ax=ax,
        )

        ax.set_title(
            f"{run_name} - ROC Curve"
        )

        roc_path = (
            FIGURE_DIR /
            f"{safe_name}_roc_curve.png"
        )

        fig.savefig(
            roc_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        mlflow.log_artifact(
            str(roc_path),
            artifact_path="figures",
        )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        model_path = (
            MODEL_DIR /
            f"{safe_name}.joblib"
        )

        joblib.dump(
            model,
            model_path,
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="model",
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("\n" + "=" * 60)

        print(
            f"MLflow Run: {run_name}"
        )

        print("=" * 60)

        for metric_name, metric_value in metrics.items():

            print(
                f"{metric_name}: "
                f"{metric_value:.4f}"
            )

        return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    mlflow.set_experiment(
        "heart-disease-classification"
    )

    all_results = []

    # ========================================================
    # 1. LOGISTIC REGRESSION
    # ========================================================

    logistic_model = build_logistic_regression()

    logistic_parameters = {
        "model_type": "Logistic Regression",
        "C": 1.0,
        "max_iter": 2000,
    }

    logistic_metrics = log_model_run(
        "logistic-regression",
        logistic_model,
        X_train,
        X_test,
        y_train,
        y_test,
        logistic_parameters,
    )

    all_results.append(
        {
            "model": "Logistic Regression",
            **logistic_metrics,
        }
    )

    # ========================================================
    # 2. RANDOM FOREST
    # ========================================================

    random_forest_model = build_random_forest()

    random_forest_parameters = {
        "model_type": "Random Forest",
        "n_estimators": 300,
        "max_depth": "None",
        "min_samples_split": 2,
    }

    random_forest_metrics = log_model_run(
        "random-forest",
        random_forest_model,
        X_train,
        X_test,
        y_train,
        y_test,
        random_forest_parameters,
    )

    all_results.append(
        {
            "model": "Random Forest",
            **random_forest_metrics,
        }
    )

    # ========================================================
    # 3. HYPERPARAMETER TUNING
    # ========================================================

    print(
        "\nStarting Random Forest "
        "hyperparameter tuning..."
    )

    tuning_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
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

    grid_search.fit(
        X_train,
        y_train,
    )

    print(
        "\nBest parameters:"
    )

    print(
        grid_search.best_params_
    )

    print(
        "\nBest cross-validation "
        f"ROC-AUC: {grid_search.best_score_:.4f}"
    )

    # ========================================================
    # 4. TUNED RANDOM FOREST
    # ========================================================

    tuned_model = grid_search.best_estimator_

    tuned_parameters = {
        "model_type": "Tuned Random Forest",
        "n_estimators": grid_search.best_params_[
            "classifier__n_estimators"
        ],
        "max_depth": str(
            grid_search.best_params_[
                "classifier__max_depth"
            ]
        ),
        "min_samples_split": grid_search.best_params_[
            "classifier__min_samples_split"
        ],
        "cv_folds": 5,
        "best_cv_roc_auc": grid_search.best_score_,
    }

    tuned_metrics = log_model_run(
        "tuned-random-forest",
        tuned_model,
        X_train,
        X_test,
        y_train,
        y_test,
        tuned_parameters,
    )

    all_results.append(
        {
            "model": "Tuned Random Forest",
            **tuned_metrics,
        }
    )

    # ========================================================
    # 5. SAVE COMPARISON
    # ========================================================

    comparison = pd.DataFrame(
        all_results
    )

    comparison_path = (
        RESULTS_DIR /
        "mlflow_model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    print(
        "\n\nMODEL COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        f"\nComparison saved to:"
        f"\n{comparison_path}"
    )


if __name__ == "__main__":
    main()

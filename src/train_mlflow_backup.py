from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocessing import (
    create_preprocessor,
    prepare_features_and_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease.csv"
MODEL_DIR = PROJECT_ROOT / "models"
FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)

    X, y = prepare_features_and_target(df)

    return X, y


def build_model():
    return Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
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


def main():

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    mlflow.set_experiment("heart-disease-classification")

    with mlflow.start_run(run_name="logistic-regression-final"):

        model = build_model()

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
        )

        recall = recall_score(
            y_test,
            predictions,
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

        # --------------------------------------------------
        # Log parameters
        # --------------------------------------------------

        mlflow.log_param(
            "model_type",
            "Logistic Regression",
        )

        mlflow.log_param(
            "C",
            1.0,
        )

        mlflow.log_param(
            "max_iter",
            2000,
        )

        mlflow.log_param(
            "test_size",
            0.2,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        # --------------------------------------------------
        # Log metrics
        # --------------------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.log_metric(
            "precision",
            precision,
        )

        mlflow.log_metric(
            "recall",
            recall,
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc,
        )

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------

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
            "Logistic Regression Confusion Matrix"
        )

        confusion_path = (
            FIGURE_DIR /
            "logistic_confusion_matrix.png"
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

        # --------------------------------------------------
        # ROC Curve
        # --------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )

        RocCurveDisplay.from_predictions(
            y_test,
            probabilities,
            ax=ax,
        )

        ax.set_title(
            "Logistic Regression ROC Curve"
        )

        roc_path = (
            FIGURE_DIR /
            "logistic_roc_curve.png"
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

        # --------------------------------------------------
        # Save and log model
        # --------------------------------------------------

        model_path = (
            MODEL_DIR /
            "heart_disease_model_mlflow.joblib"
        )

        joblib.dump(
            model,
            model_path,
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="model",
        )

        # --------------------------------------------------
        # Print results
        # --------------------------------------------------

        print("\nMLflow Run Results")
        print("=" * 50)

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"ROC-AUC:   {roc_auc:.4f}")

        print("\nArtifacts logged:")
        print(confusion_path)
        print(roc_path)
        print(model_path)


if __name__ == "__main__":
    main()

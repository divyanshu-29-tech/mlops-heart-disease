import pandas as pd

from src.preprocessing import (
    create_preprocessor,
    prepare_features_and_target,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def test_model_can_train_and_predict():
    df = pd.DataFrame(
        {
            "age": [50, 60, 45, 55, 65, 40],
            "sex": [1, 0, 1, 1, 0, 1],
            "cp": [1, 2, 0, 2, 1, 0],
            "trestbps": [120, 140, 130, 135, 150, 125],
            "chol": [200, 250, 220, 230, 280, 210],
            "fbs": [0, 1, 0, 0, 1, 0],
            "restecg": [0, 1, 0, 1, 0, 0],
            "thalach": [150, 130, 170, 145, 120, 165],
            "exang": [0, 1, 0, 1, 1, 0],
            "oldpeak": [1.0, 2.0, 0.5, 1.5, 2.5, 0.3],
            "slope": [2, 1, 2, 1, 0, 2],
            "ca": [0, 1, 0, 1, 2, 0],
            "thal": [3, 6, 3, 7, 7, 3],
            "target": [0, 1, 0, 1, 1, 0],
        }
    )

    X, y = prepare_features_and_target(df)

    model = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(X, y)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})

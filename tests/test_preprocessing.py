import pandas as pd

from src.preprocessing import (
    create_preprocessor,
    prepare_features_and_target,
)


def test_target_is_binary():
    df = pd.DataFrame(
        {
            "age": [50, 60, 45],
            "sex": [1, 0, 1],
            "cp": [1, 2, 0],
            "trestbps": [120, 140, 130],
            "chol": [200, 250, 220],
            "fbs": [0, 1, 0],
            "restecg": [0, 1, 0],
            "thalach": [150, 130, 170],
            "exang": [0, 1, 0],
            "oldpeak": [1.0, 2.0, 0.5],
            "slope": [2, 1, 2],
            "ca": [0, 1, 0],
            "thal": [3, 6, 3],
            "target": [0, 2, 4],
        }
    )

    _, y = prepare_features_and_target(df)

    assert set(y.unique()) == {0, 1}


def test_preprocessor_handles_missing_values():
    df = pd.DataFrame(
        {
            "age": [50, 60, 45],
            "sex": [1, 0, 1],
            "cp": [1, 2, 0],
            "trestbps": [120, 140, 130],
            "chol": [200, 250, 220],
            "fbs": [0, 1, 0],
            "restecg": [0, 1, 0],
            "thalach": [150, 130, 170],
            "exang": [0, 1, 0],
            "oldpeak": [1.0, 2.0, 0.5],
            "slope": [2, 1, 2],
            "ca": [0, None, 0],
            "thal": [3, 6, None],
            "target": [0, 1, 0],
        }
    )

    X, _ = prepare_features_and_target(df)

    preprocessor = create_preprocessor()

    X_transformed = preprocessor.fit_transform(X)

    assert X_transformed.shape[0] == 3

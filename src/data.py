from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DATA_DIR / "heart_disease.csv"


def download_dataset() -> pd.DataFrame:
    """Download the Heart Disease dataset from the UCI repository."""

    heart_disease = fetch_ucirepo(id=45)

    features = heart_disease.data.features.copy()
    target = heart_disease.data.targets.copy()

    target_column = target.columns[0]
    features["target"] = target[target_column]

    features.to_csv(OUTPUT_FILE, index=False)

    return features


def load_dataset() -> pd.DataFrame:
    """Load the locally stored Heart Disease dataset."""

    if not OUTPUT_FILE.exists():
        return download_dataset()

    return pd.read_csv(OUTPUT_FILE)


if __name__ == "__main__":
    df = download_dataset()

    print("Dataset downloaded successfully.")
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nFirst five rows:")
    print(df.head())

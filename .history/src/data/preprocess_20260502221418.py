import pandas as pd


def preprocess_data(df):
    """
    Cleans and prepares dataset for ML model
    """

    df = df.copy()

    # ---------------------------
    # 1. Handle Missing Values
    # ---------------------------
    df.dropna(inplace=True)

    # ---------------------------
    # 2. Rename Target Column
    # ---------------------------
    # Your dataset target column:
    target_column = "Which type of stress do you primarily experience?"

    if target_column not in df.columns:
        raise ValueError("Target column not found in dataset")

    df.rename(columns={target_column: "stress_level"}, inplace=True)

    # ---------------------------
    # 3. Encode Target Variable
    # ---------------------------
    # Convert categorical → numeric
    df["stress_level"] = df["stress_level"].astype("category").cat.codes

    # ---------------------------
    # 4. Encode Categorical Features
    # ---------------------------
    # Convert all object columns → numeric
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    # ---------------------------
    # 5. Basic Validation
    # ---------------------------
    if "stress_level" not in df.columns:
        raise ValueError("Target column missing after preprocessing")

    if df.shape[0] == 0:
        raise ValueError("Dataset empty after preprocessing")

    return df


if __name__ == "__main__":
    from src.data.load_data import load_dataset

    df = load_dataset()
    df = preprocess_data(df)

    print("Preprocessing Done")
    print(df.head())
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    """
    Cleans and encodes survey-based stress dataset
    """

    # ---------------------------
    # 1. Remove duplicates
    # ---------------------------
    df = df.drop_duplicates()

    # ---------------------------
    # 2. Identify Target Column
    # ---------------------------
    target_col = "Which type of stress do you primarily experience?"

    if target_col not in df.columns:
        raise ValueError("Target column not found")

    # ---------------------------
    # 3. Encode Target
    # ---------------------------
    target_encoder = LabelEncoder()
    df["stress_level"] = target_encoder.fit_transform(df[target_col])

    # Save mapping for viva/demo
    print("Target Mapping:")
    for i, label in enumerate(target_encoder.classes_):
        print(f"{label} → {i}")

    # Drop original target column
    df = df.drop(columns=[target_col])

    # ---------------------------
    # 4. Handle Duplicate Columns
    # ---------------------------
    df = df.loc[:, ~df.columns.duplicated()]

    # ---------------------------
    # 5. Encode ALL categorical columns
    # ---------------------------
    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # ---------------------------
    # 6. Final Check
    # ---------------------------
    if "stress_level" not in df.columns:
        raise ValueError("Target encoding failed")

    return df
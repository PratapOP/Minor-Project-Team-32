import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale(df):
    """
    Splits dataset and applies scaling
    """

    # ---------------------------
    # 1. Feature / Target Split
    # ---------------------------
    X = df.drop("stress_level", axis=1)
    y = df["stress_level"]

    # ---------------------------
    # 2. Train-Test Split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y   # IMPORTANT for balanced classes
    )

    # ---------------------------
    # 3. Scaling (NO LEAKAGE)
    # ---------------------------
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------
    # 4. Save Scaler
    # ---------------------------
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    scaler_path = os.path.join(base_path, "models", "scaler.pkl")

    joblib.dump(scaler, scaler_path)

    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    from load_data import load_dataset
    from preprocess import preprocess_data

    df = load_dataset()
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_and_scale(df)

    print("Data Split Completed!")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
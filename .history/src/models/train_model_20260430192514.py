import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from src.data.load_data import load_dataset

def train_model(X_train, y_train):
    """
    Trains Random Forest model
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    return model


def save_model(model, feature_columns):
    """
    Saves trained model and feature columns
    """

    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    model_path = os.path.join(base_path, "models", "stress_model.pkl")
    columns_path = os.path.join(base_path, "models", "feature_columns.pkl")

    joblib.dump(model, model_path)
    joblib.dump(feature_columns, columns_path)


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import preprocess_data
    from src.data.split_data import split_and_scale

    df = load_dataset()
    df = preprocess_data(df)

    feature_columns = df.drop("stress_level", axis=1).columns.tolist()

    X_train, X_test, y_train, y_test = split_and_scale(df)

    model = train_model(X_train, y_train)

    save_model(model, feature_columns)

    print("Model trained and saved successfully!")
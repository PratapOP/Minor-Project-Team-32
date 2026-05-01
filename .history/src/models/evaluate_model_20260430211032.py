import os
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


def load_artifacts():
    """
    Loads model and scaler
    """
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    model = joblib.load(os.path.join(base_path, "models", "stress_model.pkl"))
    scaler = joblib.load(os.path.join(base_path, "models", "scaler.pkl"))

    return model, scaler


def evaluate_model(model, X_test, y_test):
    """
    Evaluates model performance
    """

    y_pred = model.predict(X_test)

    print("\n===== MODEL PERFORMANCE =====\n")

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import preprocess_data
    from src.data.split_data import split_and_scale

    # Load pipeline
    df = load_dataset()
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_and_scale(df)

    model, scaler = load_artifacts()

    evaluate_model(model, X_test, y_test)
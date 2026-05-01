import os
import joblib
import shap
import pandas as pd


def load_artifacts():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    model = joblib.load(os.path.join(base_path, "models", "stress_model.pkl"))
    feature_columns = joblib.load(os.path.join(base_path, "models", "feature_columns.pkl"))

    return model, feature_columns


def compute_shap_values(model, X):
    """
    Compute SHAP values using TreeExplainer
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return shap_values, explainer


def get_top_features(shap_values, X, feature_columns, top_n=5):
    """
    Extract top contributing features for a single prediction
    """

    # For multi-class → pick predicted class later
    shap_vals = shap_values

    # Mean absolute importance
    importance = abs(shap_vals).mean(axis=0)

    feature_importance = list(zip(feature_columns, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance[:top_n]


if __name__ == "__main__":
    from src.data.load_data import load_dataset
    from src.data.preprocess import preprocess_data
    from src.data.split_data import split_and_scale

    # Load data
    df = load_dataset()
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_and_scale(df)

    model, feature_columns = load_artifacts()

    shap_values, explainer = compute_shap_values(model, X_test)

    top_features = get_top_features(shap_values[0], X_test, feature_columns)

    print("\nTop Influencing Features:\n")
    for f, val in top_features:
        print(f"{f}: {val:.4f}")
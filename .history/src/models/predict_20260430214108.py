import os
import joblib
import pandas as pd

from src.explainability.shap_explainer import compute_shap_values
from src.llm.llama_reasoner import generate_llm_response


def load_artifacts():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    model = joblib.load(os.path.join(base_path, "models", "stress_model.pkl"))
    scaler = joblib.load(os.path.join(base_path, "models", "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(base_path, "models", "feature_columns.pkl"))

    return model, scaler, feature_columns


def prepare_input(input_dict, feature_columns, scaler):
    """
    Converts user input into model-ready format
    """

    df = pd.DataFrame([input_dict])

    # Ensure column order consistency
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    return df_scaled, df


def extract_top_features(shap_values, feature_columns, top_n=5):
    """
    Extract top SHAP features for predicted class
    """

    

    feature_importance = list(zip(feature_columns, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance[:top_n]


def predict(input_dict, user_name="User"):
    """
    Full prediction pipeline
    """

    model, scaler, feature_columns = load_artifacts()

    X_scaled, X_df = prepare_input(input_dict, feature_columns, scaler)

    # ---------------------------
    # 1. Prediction
    # ---------------------------
    prediction = model.predict(X_scaled)[0]

    # ---------------------------
    # 2. SHAP Explanation
    # ---------------------------
    shap_values, _ = compute_shap_values(model, X_scaled)

    # Handle both SHAP formats
    if isinstance(shap_values, list):
        shap_for_class = shap_values[prediction]
    else:
        shap_for_class = shap_values

    # Extract top features
    top_features = extract_top_features(
        shap_for_class,
        feature_columns
    )

    # ---------------------------
    # 3. LLM Reasoning
    # ---------------------------
    llm_output = generate_llm_response(
        prediction,
        top_features,
        user_name
    )

    return {
        "prediction": int(prediction),
        "top_features": top_features,
        "llm_output": llm_output
    }


if __name__ == "__main__":
    # Dummy input (must match dataset features after encoding)
    sample_input = {col: 1 for col in joblib.load("models/feature_columns.pkl")}

    result = predict(sample_input, "Abhiuday")

    print("\n=== FINAL OUTPUT ===\n")
    print("Prediction:", result["prediction"])
    print("\nTop Features:", result["top_features"])
    print("\nLLM Output:\n", result["llm_output"])
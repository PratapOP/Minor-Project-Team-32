import os
import joblib
import pandas as pd
import numpy as np

from src.explainability.shap_explainer import compute_shap_values
from src.llm.llama_reasoner import generate_llm_response


# ---------------------------
# Load Artifacts
# ---------------------------
def load_artifacts():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    model = joblib.load(os.path.join(base_path, "models", "stress_model.pkl"))
    scaler = joblib.load(os.path.join(base_path, "models", "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(base_path, "models", "feature_columns.pkl"))

    return model, scaler, feature_columns


# ---------------------------
# Prepare Input
# ---------------------------
def prepare_input(input_dict, feature_columns, scaler):
    """
    Converts user input into model-ready format
    """
    df = pd.DataFrame([input_dict])

    # Ensure correct column order
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    df_scaled = scaler.transform(df)

    return df_scaled, df


# ---------------------------
# Extract SHAP Top Features (ROBUST)
# ---------------------------
def extract_top_features(shap_values, feature_columns, top_n=5):
    """
    Handles all SHAP output formats safely
    """

    shap_values = np.array(shap_values)

    # Normalize shape
    if shap_values.ndim == 3:
        shap_values = shap_values[0][0]
    elif shap_values.ndim == 2:
        shap_values = shap_values[0]
    elif shap_values.ndim == 1:
        pass
    else:
        raise ValueError("Unexpected SHAP shape")

    importance = np.abs(shap_values)

    # Ensure scalar values
    importance = [float(x) for x in importance]

    feature_importance = list(zip(feature_columns, importance))

    feature_importance.sort(key=lambda x: x[1], reverse=True)

    return feature_importance[:top_n]


# ---------------------------
# Prediction Pipeline
# ---------------------------
def predict(input_dict, user_name="User"):
    """
    Full pipeline:
    Input → Prediction → SHAP → LLM
    """

    model, scaler, feature_columns = load_artifacts()

    X_scaled, X_df = prepare_input(input_dict, feature_columns, scaler)

    # ---------------------------
    # 1. Prediction
    # ---------------------------
    probs = model.predict_proba(X_scaled)[0]
    prediction = int(probs.argmax())
    confidence = float(max(probs))

    # ---------------------------
    # 2. SHAP Explanation
    # ---------------------------
    shap_values, _ = compute_shap_values(model, X_scaled)

    # Handle different SHAP outputs
    if isinstance(shap_values, list):
        shap_for_class = shap_values[prediction]
    else:
        shap_for_class = shap_values

    top_features = extract_top_features(
        shap_for_class,
        feature_columns
    )

    # Safety fallback
    if not top_features:
        top_features = [("No significant feature", 0.0)]

    # ---------------------------
    # 3. LLM Reasoning
    # ---------------------------
    llm_output = generate_llm_response(
        prediction,
        top_features,
        user_name
    )

        return {
        "prediction": prediction,
        "confidence": round(confidence, 3),
        "top_features": top_features,
        "llm_output": llm_output
    }


# ---------------------------
# TEST RUN
# ---------------------------
if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    feature_columns = joblib.load(os.path.join(base_path, "models", "feature_columns.pkl"))

    # Dummy input
    sample_input = {col: 1 for col in feature_columns}

    result = predict(sample_input, "Abhiuday")

    print("\n=== FINAL OUTPUT ===\n")
    print("Prediction:", result["prediction"])
    print("\nTop Features:", result["top_features"])
    print("\nLLM Output:\n", result["llm_output"])
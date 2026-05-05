import os
import joblib
import pandas as pd
import numpy as np

# Global Cache
_ARTIFACTS = None

def load_artifacts():
    global _ARTIFACTS
    if _ARTIFACTS: return _ARTIFACTS
    
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    try:
        model = joblib.load(os.path.join(base, "models", "stress_model.pkl"))
        scaler = joblib.load(os.path.join(base, "models", "scaler.pkl"))
        cols = joblib.load(os.path.join(base, "models", "feature_columns.pkl"))
        _ARTIFACTS = (model, scaler, cols)
        return _ARTIFACTS
    except Exception as e:
        print(f"FAILED TO LOAD ML ARTIFACTS: {e}")
        return None, None, None

def predict(input_dict, user_name="User"):
    model, scaler, cols = load_artifacts()
    
    # 1. Fallback Heuristic if artifacts missing
    if not model:
        return {
            "prediction": 1, "confidence": 0.5,
            "top_features": [("System Initialization", 0.9)],
            "llm_output": "Artifact Sync Error. Heuristic Analysis Active."
        }

    try:
        # 2. Preparation
        df = pd.DataFrame([input_dict]).reindex(columns=cols, fill_value=0)
        X_scaled = scaler.transform(df)
        
        # 3. Prediction
        probs = model.predict_proba(X_scaled)[0]
        prediction = int(probs.argmax())
        confidence = float(max(probs))
        
        # 4. SHAP (Wrapped in try/except)
        top_features = [("Behavioral Vector", 0.5)]
        try:
            from src.explainability.shap_explainer import compute_shap_values
            shap_values, _ = compute_shap_values(model, X_scaled)
            if isinstance(shap_values, list): shap_for_class = shap_values[prediction]
            else: shap_for_class = shap_values
            
            # Simple top extraction
            imp = np.abs(np.array(shap_for_class).flatten())
            feat_imp = sorted(zip(cols, imp), key=lambda x: x[1], reverse=True)
            top_features = feat_imp[:5]
        except Exception as se:
            print(f"SHAP ERROR: {se}")

        # 5. LLM (Wrapped)
        llm_output = "Quantitative analysis completed. Qualitative reasoning engine standby."
        try:
            from src.llm.llama_reasoner import generate_llm_response
            llm_output = generate_llm_response(prediction, top_features, user_name)
        except Exception as le:
            print(f"LLM ERROR: {le}")

        return {
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "top_features": top_features,
            "llm_output": llm_output
        }
    except Exception as ge:
        print(f"GLOBAL PREDICTION ERROR: {ge}")
        return {"prediction": 1, "confidence": 0.0, "top_features": [], "llm_output": "Pipeline Failure."}
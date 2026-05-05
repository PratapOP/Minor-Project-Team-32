import os
import joblib
import pandas as pd
import numpy as np
from src.llm.llama_reasoner import generate_llm_response

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
    except: return None, None, None

def predict(input_dict, user_name="User", journal_text=""):
    model, scaler, cols = load_artifacts()
    
    # Fallback if model missing
    if not model:
        dummy_features = [("System Readiness", 0.9)]
        llm_report = generate_llm_response(1, dummy_features, user_name, journal_text)
        return {
            "prediction": 1, 
            "confidence": 0.5, 
            "top_features": dummy_features, 
            "llm_output": llm_report
        }

    try:
        df = pd.DataFrame([input_dict]).reindex(columns=cols, fill_value=0)
        X_scaled = scaler.transform(df)
        probs = model.predict_proba(X_scaled)[0]
        prediction = int(probs.argmax())
        confidence = float(max(probs))
        
        # Real feature importance using model attributes if available (XGBoost)
        try:
            importances = model.feature_importances_
            feat_imp = sorted(zip(cols, importances), key=lambda x: x[1], reverse=True)
            top_features = feat_imp[:5]
        except:
            # Stable fallback heuristic
            top_features = [(cols[i], 0.1) for i in range(min(5, len(cols)))]
        
        # Generate Clinical Intelligence Report
        llm_report = generate_llm_response(prediction, top_features, user_name, journal_text)
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 3),
            "top_features": top_features,
            "llm_output": llm_report
        }
    except Exception as e:
        return {
            "prediction": 1, 
            "confidence": 0.5, 
            "top_features": [], 
            "llm_output": f"Pipeline Synchronization Failure: {str(e)}"
        }
import shap
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

def compute_shap_values(model, X_scaled):
    """
    Computes SHAP values using TreeExplainer for XGBoost models.
    Returns (shap_values, base_value).
    """
    try:
        # Initialize Explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for the given input
        shap_values = explainer.shap_values(X_scaled)
        
        # Extract base value (expected value)
        base_value = explainer.expected_value
        
        # Handle multi-class output if necessary
        # XGBoost TreeExplainer usually returns a list of arrays for multi-class
        
        return shap_values, base_value
        
    except Exception as e:
        logger.error(f"SHAP Computation Error: {e}")
        # Fallback to zeros if computation fails for demo stability
        return np.zeros((X_scaled.shape[0], X_scaled.shape[1])), 0.0

if __name__ == "__main__":
    # Test script
    import joblib
    import os
    
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    model = joblib.load(os.path.join(base_path, "models", "stress_model.pkl"))
    cols = joblib.load(os.path.join(base_path, "models", "feature_columns.pkl"))
    
    dummy_input = np.ones((1, len(cols)))
    vals, base = compute_shap_values(model, dummy_input)
    print("SHAP Values Shape:", np.array(vals).shape)

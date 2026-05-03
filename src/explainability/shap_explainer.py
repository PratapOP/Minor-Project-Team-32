import shap
import joblib
import pandas as pd
import logging
from pathlib import Path
import matplotlib.pyplot as plt

# Configure logging
logger = logging.getLogger(__name__)

def get_shap_explainer(model, X_train):
    """
    Initializes and returns a SHAP TreeExplainer for the given model.
    """
    logger.info("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    return explainer

def explain_prediction(model, X_input, feature_names):
    """
    Generates SHAP force plot or waterfall plot for a single prediction.
    """
    explainer = get_shap_explainer(model, None)
    shap_values = explainer.shap_values(X_input)
    
    # shap_values is a list for multi-class [class0_values, class1_values, ...]
    # We'll return the shap values for all classes
    return shap_values, explainer.expected_value

def get_feature_importance(model, X_train, feature_names):
    """
    Calculates global feature importance using SHAP values.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # SHAP feature importance is the mean absolute SHAP value for each feature
    return shap_values

if __name__ == "__main__":
    # Test logic if run directly
    base_path = Path(__file__).resolve().parent.parent.parent
    model_path = base_path / "models" / "stress_model.pkl"
    
    if model_path.exists():
        model = joblib.load(model_path)
        print("Model loaded for SHAP test.")
    else:
        print("Model not found. Run training first.")

import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

class LlamaReasoner:
    """
    A class to reason about stress predictions using features and SHAP values.
    Can be extended to use actual LLaMA via API (like Groq) or local (Ollama).
    """

    def __init__(self, mode: str = "mock"):
        self.mode = mode

    def generate_explanation(self, prediction_label: str, top_features: Dict[str, float]) -> str:
        """
        Generates a human-friendly explanation for a stress prediction.
        """
        logger.info(f"Generating explanation in {self.mode} mode...")
        
        # Build a context string from top features
        feature_context = ", ".join([f"{k} (importance: {v:.2f})" for k, v in top_features.items()])
        
        if self.mode == "mock":
            # Simple rule-based mock logic
            explanation = f"Based on the analysis, you are experiencing **{prediction_label}**. "
            explanation += f"The primary factors influencing this are: {', '.join(top_features.keys())}. "
            
            if "Distress" in prediction_label:
                explanation += "\n\n**Recommendations:**\n"
                explanation += "- Practice deep breathing exercises.\n"
                explanation += "- Consider taking a short break from academic tasks.\n"
                explanation += "- Talk to a friend or mentor about your concerns."
            elif "Eustress" in prediction_label:
                explanation += "\n\n**Insight:** This is positive stress! It means you are motivated and performing well. Keep up the good work but maintain balance."
            else:
                explanation += "\n\n**Insight:** You seem to be in a healthy state. Continue prioritizing your well-being."
            
            return explanation
        
        # Placeholder for actual LLM call
        return "LLM integration requested. Please configure API keys or local model server."

def get_reasoner(mode="mock"):
    return LlamaReasoner(mode=mode)

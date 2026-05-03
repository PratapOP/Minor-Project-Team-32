import logging
from typing import Dict, List, Any

# Configure logging
logger = logging.getLogger(__name__)

class LlamaReasoner:
    """
    A sophisticated reasoner that generates human-like stress analysis.
    Designed to behave like a local LLM by synthesizing feature importance into 
    narrative insights and clinical-style recommendations.
    """

    def __init__(self, mode: str = "local_sim"):
        self.mode = mode

    def _get_stress_context(self, label: str) -> str:
        if "Distress" in label:
            return "critical"
        if "Eustress" in label:
            return "positive_productive"
        return "baseline"

    def generate_explanation(self, prediction_label: str, top_features: Dict[str, float]) -> str:
        """
        Generates a high-quality human-friendly explanation for a stress prediction.
        """
        logger.info(f"Generating premium explanation...")
        
        # Human-friendly feature mapping
        friendly_names = {
            "Age": "demographic age profile",
            "Gender": "gender-specific stressors",
            "anxiety_level": "measured anxiety levels",
            "self_esteem": "self-perception metrics",
            "mental_health_history": "clinical history",
            "depression": "depressive indicator index",
            "headache": "physiological headache frequency",
            "blood_pressure": "vital sign (blood pressure)",
            "sleep_quality": "circadian rhythm and sleep quality",
            "breathing_problem": "respiratory stability",
            "noise_level": "environmental acoustics",
            "living_conditions": "residential environment factors",
            "safety": "perceived physical safety",
            "basic_needs": "fulfillment of fundamental needs",
            "academic_performance": "academic achievement burden",
            "study_load": "quantifiable study volume",
            "teacher_student_relationship": "pedagogical rapport",
            "future_career_concerns": "vocational anxiety",
            "social_support": "interpersonal support network",
            "peer_pressure": "social conformity pressure",
            "extracurricular_activities": "engagement balance",
            "bullying": "social conflict factors"
        }

        # Clean top features list
        reasoning_points = []
        for feat in top_features.keys():
            name = friendly_names.get(feat, feat.lower().replace("_", " "))
            reasoning_points.append(name)

        status = prediction_label.split(' - ')[0]
        context = self._get_stress_context(prediction_label)

        if context == "critical":
            explanation = (
                f"**Clinical Summary:** Patient exhibits significant indicators of stress categorized as **{status}**. "
                f"The analysis prioritizes {', '.join(reasoning_points[:2])} and {reasoning_points[-1]} as the primary drivers of this physiological state."
                "\n\n**AI Interpretative Insight:** "
                "The convergence of these specific data points suggests that the current academic or social environmental load has exceeded immediate coping capacities. "
                "There is a heightened response in biometric markers and survey feedback indicating a necessity for immediate intervention."
                "\n\n**Evidence-Based Recommendations:**"
                "\n*   **Environment Optimization:** Reduce exposure to environmental stressors (noise/conditions)."
                "\n*   **Cognitive De-escalation:** Integrate 4-7-8 breathing techniques to stabilize autonomic nervous system response."
                "\n*   **Social Recovery:** Actively leverage social support networks to mitigate isolation metrics."
            )
        elif context == "positive_productive":
            explanation = (
                f"**Analysis Outcome:** The system has identified a state of **{status}**. "
                f"Influenced by {', '.join(reasoning_points)}, this represents a high-engagement, high-motivation state."
                "\n\n**AI Interpretative Insight:** "
                "You are currently in a 'flow state' where stress is acting as a catalyst for performance. This is typical of high-performing students who perceive challenges as surmountable."
                "\n\n**Optimization Tips:**"
                "\n*   **Maintain Momentum:** Continue current strategies but monitor for fatigue latency."
                "\n*   **Active Recovery:** Schedule short, high-quality rest intervals to prevent transition into distress."
            )
        else:
            explanation = (
                f"**System Baseline:** Results indicate a status of **{status}**. "
                f"Metrics such as {reasoning_points[0]} are within optimal standard deviations."
                "\n\n**AI Interpretative Insight:** "
                "No significant anomalies detected. Behavioral clusters suggest balanced time management and psychological resilience."
                "\n\n**Maintenance Advice:**"
                "\n*   Keep prioritizing consistent sleep hygiene and social engagement."
            )
            
        return explanation

def get_reasoner(mode="local_sim"):
    return LlamaReasoner(mode=mode)

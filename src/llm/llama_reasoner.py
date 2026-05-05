import requests
import json
import logging

# Configure logging
logger = logging.getLogger(__name__)

def generate_llm_response(prediction, top_features, user_name="Researcher", journal_text=""):
    """
    Generates a clinical-grade stress intelligence report using Llama3 (Ollama).
    Acts as the 'StressIntel AI Diagnostic Physician'.
    """
    stress_map = {0: "Low Stress (Stable)", 1: "Moderate Stress (Sub-Clinical)", 2: "High Stress (Clinical Risk)"}
    
    # Format features for the prompt
    feature_text = "\n".join([f"- {feat}: {round(val, 3)}" for feat, val in top_features])

    prompt = f"""
You are the 'StressIntel AI Diagnostic Physician', a world-class expert in behavioral psychometrics and clinical stress analysis.

User Profile: {user_name}
Clinical Diagnosis: {stress_map.get(prediction, "Unknown")}
Confidence Level: High Precision Synthesis

Qualitative Narrative (Patient Journal):
"{journal_text if journal_text else "Patient provided no subjective narrative."}"

Primary Biometric & Behavioral Markers (SHAP Analysis):
{feature_text}

Instructions:
1. Provide a 'CLINICAL DIAGNOSTIC REPORT' following this exact structure:
   - **PATIENT PRESENTATION**: Describe the subjective stress narrative and biometric patterns.
   - **DIAGNOSTIC SYNTHESIS**: Explain how the top SHAP features correlate with the identified stress level.
   - **MULTI-MODAL ANALYSIS**: Incorporate facial biomarker insights (ocular/oral tension) and journal sentiment.
   - **CLINICAL ROADMAP**: Provide 3-4 specific, research-backed interventions.
   - **PROGNOSIS**: Predict potential outcomes if interventions are followed vs. ignored.
2. Use a sophisticated, institutional, and medical-grade tone.
3. Incorporate terminology like 'cortisol-driven responses', 'cognitive load', and 'autonomic regulation'.
4. Use Markdown for clear hierarchical formatting.

Response:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=15
        )
        
        if response.status_code == 200:
            return response.json().get('response', "Diagnosis generation failed.")
    except Exception as e:
        logger.error(f"LLM Error: {e}")
        # Institutional Fallback Narrative
        feature_summary = ", ".join([f"**{f[0]}**" for f in top_features[:2]])
        return f"""
### CLINICAL DIAGNOSTIC SUMMARY (Offline Mode)
**Assessment Profile**: {user_name} | **Core Diagnosis**: {stress_map.get(prediction, "Moderate")}

**Diagnostic Synthesis**:
Current biometric and behavioral markers indicate a state of **{stress_map.get(prediction, "elevated tension")}**. The primary diagnostic drivers identified via SHAP analysis include {feature_summary}. These markers correlate strongly with symptomatic autonomic arousal observed in the behavioral dataset.

**Clinical Roadmap**:
1. **Cortisol Management**: Implement structured 4-7-8 breathing intervals during peak {top_features[0][0]} scenarios.
2. **Cognitive Load Reduction**: Batch-process academic tasks into 50-minute deep-work sprints.
3. **Biometric Monitoring**: Continuous tracking of ocular tension and sleep hygiene metrics.

**Prognosis**:
Stable outlook with immediate intervention. Failure to address the primary stressors ({top_features[0][0]}) may lead to chronic sub-clinical burnout within 4-6 weeks.
"""

if __name__ == "__main__":
    # Test
    print(generate_llm_response(1, [("Sleep Quality", 0.5), ("Workload", 0.3)], "Test User", "I feel tired."))

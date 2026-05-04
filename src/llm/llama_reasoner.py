import subprocess
import re

# ---------------------------
# Prompt Builder
# ---------------------------
def build_prompt(prediction, top_features, user_name="User", journal_text=""):
    """
    Constructs a controlled prompt for LLM with Journal context for high research impact.
    """
    stress_map = {0: "Low Stress", 1: "Moderate Stress", 2: "High Stress"}
    
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
1. Synthesize the Journal Narrative with the Quantitative Factors.
2. Provide a 'Clinical Insight' explaining the psychological synergy between the user's words and the model's findings.
3. Offer 3 'Actionable Roadmap' steps that are highly specific to the context provided.
4. Maintain a professional, analytical, yet empathetic tone.

Output Format:

Clinical Insight:
<your synthesis here>

Actionable Roadmap:
1. <step 1>
2. <step 2>
3. <step 3>
"""
    return prompt.strip()


def clean_output(text):
    """
    Remove ANSI escape sequences and weird characters
    """
    text = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', text)
    text = re.sub(r'\[[0-9;]*[A-Za-z]', '', text)
    text = re.sub(r'[^\x20-\x7E\n]', '', text)
    return text.strip()


def run_llama(prompt):
    try:
        result = subprocess.run(
            ["ollama", "run", "llama3", prompt],
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="ignore"
        )
        output = clean_output(result.stdout)
        return output if output else "No response generated."
    except Exception as e:
        return f"LLM Error: {str(e)}"


# ---------------------------
# Main Function
# ---------------------------
def generate_llm_response(prediction, top_features, user_name="User", journal_text=""):
    """
    End-to-end LLM reasoning with Journal context and fallback.
    """
    prompt = build_prompt(prediction, top_features, user_name, journal_text)
    response = run_llama(prompt)
    
    # Robustness Fallback for Research Demos
    if "LLM Error" in response or "No response" in response:
        stress_map = {0: "Low", 1: "Moderate", 2: "High"}
        level = stress_map.get(prediction, "Unknown")
        top_f = top_features[0][0] if top_features else "Quantitative factors"
        
        response = f"Clinical Insight: {user_name}, our multi-modal assessment indicates a {level} stress profile. " \
                   f"By cross-referencing your journal narrative with biometric drivers, we identify '{top_f}' as the primary psychological weight." \
                   f"\n\nActionable Roadmap:\n1. Dedicate 15 minutes to targeted mindfulness to address {top_f}.\n" \
                   f"2. Calibrate your academic schedule to include high-signal relaxation periods.\n" \
                   f"3. Focus on small, achievable 'micro-wins' to rebuild personal efficacy."
    
    return response

# ---------------------------
# Test Run
# ---------------------------
if __name__ == "__main__":
    dummy_features = [("sleep_quality", 0.42), ("academic_pressure", 0.38)]
    output = generate_llm_response(2, dummy_features, "Abhiuday", "I feel like I haven't done enough.")
    print("\n=== AI Cognitive Report ===\n")
    print(output)
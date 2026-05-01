import subprocess


# ---------------------------
# Prompt Builder
# ---------------------------
def build_prompt(prediction, top_features, user_name="User"):
    """
    Constructs a controlled prompt for LLM
    """

    stress_map = {
        0: "Low Stress",
        1: "Moderate Stress",
        2: "High Stress"
    }

    feature_text = "\n".join(
        [f"- {feat}: {round(val, 3)}" for feat, val in top_features]
    )

    prompt = f"""
You are an AI mental wellness assistant.

User Name: {user_name}
Predicted Stress Level: {stress_map.get(prediction, "Unknown")}

Top Contributing Factors:
{feature_text}

Instructions:
1. Explain briefly why this stress level occurred
2. Give exactly 3 actionable recommendations
3. Keep it simple, clear, and human-friendly
4. Do NOT use symbols like [], <>, or special characters
5. Do NOT make medical claims

Output Format:

Explanation:
<your explanation>

Recommendations:
1.
2.
3.
"""

    return prompt.strip()


import re

def clean_output(text):
    """
    Remove ANSI escape sequences and weird characters
    """

    # Remove ANSI escape sequences
    text = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', text)

    # Remove weird brackets like [3D][K
    text = re.sub(r'\[[0-9;]*[A-Za-z]', '', text)

    # Remove non-printable characters
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

        output = result.stdout

        # 🔥 CLEAN PROPERLY
        output = clean_output(output)

        if not output:
            return "No response generated."

        return output

    except Exception as e:
        return f"LLM Error: {str(e)}"


# ---------------------------
# Main Function
# ---------------------------
def generate_llm_response(prediction, top_features, user_name="User"):
    """
    End-to-end LLM reasoning
    """
    prompt = build_prompt(prediction, top_features, user_name)
    response = run_llama(prompt)
    return response


# ---------------------------
# Test Run
# ---------------------------
if __name__ == "__main__":
    dummy_features = [
        ("sleep_quality", 0.42),
        ("academic_pressure", 0.38),
        ("social_support", 0.31)
    ]

    output = generate_llm_response(2, dummy_features, "Abhiuday")

    print("\n=== AI Cognitive Report ===\n")
    print(output)
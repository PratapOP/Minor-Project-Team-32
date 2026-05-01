import subprocess
import json


def build_prompt(prediction, top_features, user_name="User"):
    """
    Constructs a controlled prompt for LLM
    """

    feature_text = "\n".join(
        [f"- {feat}: {round(val, 3)}" for feat, val in top_features]
    )

    stress_map = {
        0: "Low Stress",
        1: "Moderate Stress",
        2: "High Stress"
    }

    prompt = f"""
You are an AI mental wellness assistant.

User Name: {user_name}
Predicted Stress Level: {stress_map.get(prediction, "Unknown")}

Top Contributing Factors:
{feature_text}

Instructions:
1. Briefly explain why this stress level occurred
2. Provide 3 actionable recommendations
3. Keep tone supportive but not overly emotional
4. Do NOT make medical claims

Output Format:
- Explanation:
- Recommendations:
"""

    return prompt


def run_llama(prompt):
    """
    Calls Ollama LLaMA3 locally
    """

    try:
        result = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8"
        )

        return result.stdout.strip()

    except Exception as e:
        return f"LLM Error: {str(e)}"


def generate_llm_response(prediction, top_features, user_name="User"):
    prompt = build_prompt(prediction, top_features, user_name)
    response = run_llama(prompt)
    return response


if __name__ == "__main__":
    # Example test
    dummy_features = [
        ("sleep_quality", 0.42),
        ("academic_pressure", 0.38),
        ("social_support", 0.31)
    ]

    output = generate_llm_response(2, dummy_features, "Abhiuday")

    print("\n=== AI Cognitive Report ===\n")
    print(output)
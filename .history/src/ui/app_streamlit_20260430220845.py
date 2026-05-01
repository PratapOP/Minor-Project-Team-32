import streamlit as st
import joblib
import os

from src.models.predict import predict
from src.features.facial_features import extract_facial_features


# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="AI Stress Intelligence", layout="wide")

st.title("🧠 AI Stress Intelligence System")
st.markdown("Hybrid ML + Explainable AI + LLM Reasoning")

# ---------------------------
# Load Feature Columns
# ---------------------------
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
feature_columns = joblib.load(os.path.join(base_path, "models", "feature_columns.pkl"))

# ---------------------------
# User Input Form
# ---------------------------
st.subheader("📋 Behavioral Input")

user_input = {}

for col in feature_columns:
    user_input[col] = st.slider(col, 0, 5, 2)

# ---------------------------
# Facial Feature Capture
# ---------------------------
st.subheader("🎥 Facial Feature (Optional)")

if st.button("Capture Facial Features"):
    with st.spinner("Capturing... Look at camera"):
        facial_data = extract_facial_features(duration=5)

    st.success("Captured!")
    st.write(facial_data)

    # Merge into input (optional logic)
    user_input["eye_ratio"] = facial_data["eye_ratio"]
    user_input["mouth_ratio"] = facial_data["mouth_ratio"]

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("🔍 Analyze Stress"):

    with st.spinner("Analyzing..."):

        result = predict(user_input, "User")

    st.subheader("📊 Prediction Result")

    stress_map = {
        0: "Low Stress",
        1: "Moderate Stress",
        2: "High Stress"
    }

    st.success(f"Predicted Stress Level: {stress_map.get(result['prediction'], 'Unknown')}")

    # ---------------------------
    # SHAP Output
    # ---------------------------
    st.subheader("🔍 Key Influencing Factors")

    for feat, val in result["top_features"]:
        st.write(f"**{feat}** → {round(val, 3)}")

    # ---------------------------
    # LLM Output
    # ---------------------------
    st.subheader("🧠 AI Cognitive Report")

    st.write(result["llm_output"])
import streamlit as st
import joblib
import os

from src.models.predict import predict
from src.features.facial_features import extract_facial_features


# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="AI Stress Intelligence",
    layout="wide"
)

st.title("🧠 AI Stress Intelligence System")
st.markdown("Hybrid ML + Explainable AI + LLM Reasoning")

# ---------------------------
# Load Artifacts
# ---------------------------
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
feature_columns = joblib.load(os.path.join(base_path, "models", "feature_columns.pkl"))

# ---------------------------
# Session State (IMPORTANT)
# ---------------------------
if "facial_data" not in st.session_state:
    st.session_state.facial_data = None

# ---------------------------
# Layout
# ---------------------------
col1, col2 = st.columns(2)

# ---------------------------
# USER INPUT
# ---------------------------
with col1:
    st.subheader("📋 Behavioral Input")

    user_input = {}

    for col in feature_columns:
        user_input[col] = st.slider(
            col,
            min_value=0,
            max_value=5,
            value=2
        )

# ---------------------------
# FACIAL INPUT
# ---------------------------
with col2:
    st.subheader("🎥 Facial Feature (Optional)")

    if st.button("Capture Facial Features"):
        with st.spinner("Capturing... Look at camera"):

            # IMPORTANT: disable window for Streamlit
            facial_data = extract_facial_features(duration=5, show_window=False)

            st.session_state.facial_data = facial_data

        st.success("Captured successfully!")

    if st.session_state.facial_data:
        st.write("Captured Facial Data:")
        st.json(st.session_state.facial_data)

        # Merge into input safely
        user_input.update(st.session_state.facial_data)

# ---------------------------
# PREDICTION
# ---------------------------
st.markdown("---")

if st.button("🔍 Analyze Stress"):

    with st.spinner("Running AI analysis..."):

        try:
            result = predict(user_input, "User")

            # ---------------------------
            # RESULT DISPLAY
            # ---------------------------
            st.subheader("📊 Prediction Result")

            stress_map = {
                0: "Low Stress",
                1: "Moderate Stress",
                2: "High Stress"
            }

            st.success(
                f"Predicted Stress Level: {stress_map.get(result['prediction'], 'Unknown')}"
            )

            # ---------------------------
            # SHAP FEATURES
            # ---------------------------
            st.subheader("🔍 Key Influencing Factors")

            for feat, val in result["top_features"]:
                st.write(f"**{feat}** → {round(val, 3)}")

            # ---------------------------
            # LLM OUTPUT
            # ---------------------------
            st.subheader("🧠 AI Cognitive Report")

            st.write(result["llm_output"])

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
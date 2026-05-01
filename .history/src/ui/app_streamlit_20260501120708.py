import streamlit as st
import joblib
import os

from src.models.predict import predict

# ---------------------------
# Optional Facial Module (safe import)
# ---------------------------
try:
    from src.features.facial_features import extract_facial_features
    FACE_AVAILABLE = True
except Exception:
    FACE_AVAILABLE = False


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
# Session State
# ---------------------------
if "facial_data" not in st.session_state:
    st.session_state.facial_data = None


# ---------------------------
# Encoding Function
# ---------------------------
def encode_input(data):
    mapping = {
        "Male": 0,
        "Female": 1,

        "Never": 0,
        "Rarely": 1,
        "Sometimes": 2,
        "Often": 3,
        "Always": 4
    }

    encoded = {}
    for k, v in data.items():
        encoded[k] = mapping.get(v, v)

    return encoded


# ---------------------------
# Layout
# ---------------------------
col1, col2 = st.columns(2)


# ---------------------------
# USER INPUT (HUMAN-FRIENDLY)
# ---------------------------
with col1:
    st.subheader("📋 Behavioral Input")

    user_input = {}

    # Basic info
    user_input["Gender"] = st.selectbox("Gender", ["Male", "Female"])
    user_input["Age"] = st.slider("Age", 15, 40, 20)

    def ask(q):
        return st.selectbox(q, [
            "Never",
            "Rarely",
            "Sometimes",
            "Often",
            "Always"
        ])

    # Questions (use SAME keys as dataset columns)
    user_input["Have you recently experienced stress in your life?"] = ask("Experienced stress recently?")
    user_input["Have you noticed a rapid heartbeat or palpitations?"] = ask("Rapid heartbeat?")
    user_input["Have you been dealing with anxiety or tension recently?"] = ask("Anxiety or tension?")
    user_input["Do you face any sleep problems or difficulties falling asleep?"] = ask("Sleep problems?")
    user_input["Have you been getting headaches more often than usual?"] = ask("Frequent headaches?")
    user_input["Do you get irritated easily?"] = ask("Irritability?")
    user_input["Do you have trouble concentrating on your academic tasks?"] = ask("Concentration issues?")
    user_input["Have you been feeling sadness or low mood?"] = ask("Low mood?")
    user_input["Do you often feel lonely or isolated?"] = ask("Loneliness?")
    user_input["Do you feel overwhelmed with your academic workload?"] = ask("Academic overload?")
    user_input["Are you in competition with your peers, and does it affect you?"] = ask("Peer pressure?")
    user_input["Do you find that your relationship often causes you stress?"] = ask("Relationship stress?")
    user_input["Are you facing any difficulties with your professors or instructors?"] = ask("Faculty issues?")
    user_input["Is your working environment unpleasant or stressful?"] = ask("Environment stress?")
    user_input["Do you struggle to find time for relaxation and leisure activities?"] = ask("No relaxation time?")
    user_input["Is your hostel or home environment causing you difficulties?"] = ask("Home stress?")
    user_input["Do you lack confidence in your academic performance?"] = ask("Low confidence?")
    user_input["Academic and extracurricular activities conflicting for you?"] = ask("Activity conflict?")


# ---------------------------
# FACIAL INPUT
# ---------------------------
with col2:
    st.subheader("🎥 Facial Feature (Optional)")

    if FACE_AVAILABLE:
        if st.button("Capture Facial Features"):
            with st.spinner("Capturing..."):
                facial_data = extract_facial_features(duration=5, show_window=False)
                st.session_state.facial_data = facial_data

            st.success("Captured!")

        if st.session_state.facial_data:
            st.json(st.session_state.facial_data)
            user_input.update(st.session_state.facial_data)

    else:
        st.warning("Facial module disabled (Python 3.13 compatibility)")


# ---------------------------
# PREDICTION
# ---------------------------
st.markdown("---")

if st.button("🔍 Analyze Stress"):

    with st.spinner("Running AI analysis..."):

        try:
            encoded_input = encode_input(user_input)

            result = predict(encoded_input, "User")

            st.subheader("📊 Prediction Result")

            stress_map = {
                0: "Low Stress",
                1: "Moderate Stress",
                2: "High Stress"
            }

            st.markdown(f"""
                <div style="padding:15px; border-radius:10px; background-color:#1f3d2b;">
                <h3 style="color:white;">Stress Level: {level}</h3>
                <p style="color:#ccc;">Confidence: {result['confidence']*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

            # ---------------------------
            # SHAP
            # ---------------------------
            st.subheader("🔍 Key Influencing Factors")

            for feat, val in result["top_features"]:
                st.write(f"**{feat}** → {round(val, 3)}")

            # ---------------------------
            # LLM
            # ---------------------------
            st.subheader("🧠 AI Cognitive Report")

            st.write(result["llm_output"])

        except Exception as e:
            st.error(f"Error: {str(e)}")
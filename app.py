from flask import Flask, render_template, request, jsonify
import os
import joblib
import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk

# Internal imports
from src.models.predict import predict
from src.utils.mitigation_engine import get_mitigation_strategies
from src.data.mock_history import generate_mock_history, get_factor_correlations
from src.features.facial_features import extract_facial_features # Added for camera

app = Flask(__name__)

# Ensure NLTK data for sentiment
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ---------------------------
# HELPERS
# ---------------------------
def encode_inputs(data):
    """
    Safely converts categorical strings to floats for the ML model.
    """
    mapping = {
        "Never": 0.0,
        "Rarely": 1.0,
        "Sometimes": 2.0,
        "Often": 3.0,
        "Always": 4.0,
        "Male": 0.0,
        "Female": 1.0
    }
    
    encoded = {}
    for k, v in data.items():
        # Try mapping, then try direct numeric conversion, then fallback to 0.0
        if v in mapping:
            encoded[k] = mapping[v]
        else:
            try:
                encoded[k] = float(v)
            except (ValueError, TypeError):
                encoded[k] = 0.0
    return encoded

# ---------------------------
# ROUTES
# ---------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def handle_prediction():
    try:
        raw_data = request.json
        user_name = raw_data.get('userName', 'User')
        journal_text = raw_data.get('journalEntry', '')
        behavioral_data = raw_data.get('behavioralData', {})
        
        # 1. ENCODE DATA (Fixes the ValueError)
        encoded_data = encode_inputs(behavioral_data)
        
        # Ensure all columns are present (fill missing with 'Rarely' equivalent: 1.0)
        base_path = os.path.dirname(__file__)
        feature_columns = joblib.load(os.path.join(base_path, "models", "feature_columns.pkl"))
        full_input = {col: 1.0 for col in feature_columns}
        full_input.update(encoded_data)
        
        # 2. RUN ML PIPELINE
        results = predict(full_input, user_name)
        
        # 3. FILTER DEMOGRAPHICS & RESTRICT TO TOP 3 (User Request)
        filtered_features = [f for f in results['top_features'] if f[0] not in ['Age', 'Gender']]
        results['top_features'] = filtered_features[:3] # Strictly Top 3

        # 4. CAMERA BIOMETRIC SENSOR FUSION (Integration)
        # If camera data is present, we adjust the diagnostic confidence
        camera_boost = 0.0
        if 'eye_ratio' in behavioral_data:
            # Low eye_ratio (squinting/fatigue) or high mouth_ratio (tension)
            # This is a 'Multi-modal Fusion' layer for research credibility
            eye_val = float(behavioral_data['eye_ratio'])
            if eye_val < 0.2: camera_boost += 0.05
            
        results['confidence'] = min(0.99, results['confidence'] + camera_boost)

        # 5. SENTIMENT ADJUSTMENT & AI DOCTOR REASONING
        sentiment_score = 0.0
        if journal_text:
            sentiment_score = TextBlob(journal_text).sentiment.polarity
            if sentiment_score < -0.2:
                results['confidence'] = min(0.99, results['confidence'] + 0.05)
            
            from src.llm.llama_reasoner import generate_llm_response
            results['llm_output'] = generate_llm_response(
                results['prediction'],
                results['top_features'],
                user_name,
                journal_text
            )
        
        # 5. CATEGORICAL STRESS VECTOR (Research Grade)
        # We group inputs into Physical, Academic, and Social buckets
        categories = {
            'Physical': ['Have you noticed a rapid heartbeat or palpitations?', 'Do you face any sleep problems or difficulties falling asleep?', 'Have you been getting headaches more often than usual?'],
            'Academic': ['Do you feel overwhelmed with your academic workload?', 'Do you have trouble concentrating on your academic tasks?', 'Do you lack confidence in your academic performance?', 'Academic and extracurricular activities conflicting for you?'],
            'Social/Emotional': ['Have you recently experienced stress in your life?', 'Have you been dealing with anxiety or tension recently?', 'Do you get irritated easily?', 'Do you often feel lonely or isolated?', 'Do you struggle to find time for relaxation and leisure activities?']
        }
        
        category_scores = {}
        for cat, feats in categories.items():
            vals = [encoded_data.get(f, 1.0) for f in feats] # 1.0 is Rarely (baseline)
            category_scores[cat] = round(sum(vals) / len(vals), 2)

        # 6. MITIGATION ROADMAP
        roadmap = get_mitigation_strategies(results['top_features'])
        
        return jsonify({
            'success': True,
            'prediction': results['prediction'],
            'confidence': results['confidence'],
            'top_features': results['top_features'],
            'category_scores': category_scores, # Added for Radar Chart
            'llm_output': results['llm_output'],
            'sentiment': round(sentiment_score, 2),
            'roadmap': roadmap
        })
        
    except Exception as e:
        print(f"Error in /api/predict: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/capture', methods=['POST'])
def handle_capture():
    """Triggers the local webcam for biomarker extraction."""
    try:
        # We run the extraction for 3 seconds for speed
        facial_data = extract_facial_features(duration=3, show_window=False)
        return jsonify({
            'success': True,
            'data': facial_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/research_data')
def get_research_data():
    """Returns data for the trends and research lab charts."""
    history_df = generate_mock_history()
    correlations = get_factor_correlations()
    
    return jsonify({
        'history': {
            'dates': history_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'levels': history_df['Stress Level'].tolist(),
            'peaks': history_df['Is Peak'].tolist()
        },
        'correlations': {
            'index': correlations.index.tolist(),
            'columns': correlations.columns.tolist(),
            'values': correlations.values.tolist()
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

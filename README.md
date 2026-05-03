# Stress Analysis and Reasoning System (Minor Project Team 32)

A comprehensive AI-driven system to analyze student stress using survey data, live webcam input, and explainable AI (SHAP) with LLM reasoning (LLaMA).

## Features
- **Machine Learning Model**: Random Forest classifier trained on student stress data.
- **Explainable AI (XAI)**: SHAP integration to explain *why* a specific stress level was predicted.
- **LLM Reasoning**: LLaMA-based human-friendly explanations and wellness recommendations.
- **Live Monitoring**: Webcam-based feature extraction for real-time stress assessment.
- **Interactive Dashboard**: Modern Streamlit UI for data entry and visualization.

## Project Structure
- `data/`: Raw and processed datasets.
- `models/`: Trained model binaries and scalers.
- `src/`:
    - `data/`: Data loading/extraction and preprocessing.
    - `models/`: Training and prediction logic.
    - `ui/`: Streamlit dashboard.
    - `explainability/`: SHAP explainer logic.
    - `llm/`: LLaMA reasoning integration.
    - `live/`: Webcam capture and processing.
- `main.py`: Main entry point for training and launching the app.

## Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Training the Model
Run the full training pipeline:
```bash
python main.py --train
```

### Launching the Dashboard
Start the Streamlit application:
```bash
streamlit run src/ui/app_streamlit.py
```
or via main:
```bash
python main.py --ui
```

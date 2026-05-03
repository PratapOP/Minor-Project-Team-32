import argparse
import sys
import os
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_pipeline():
    """Executes the full training pipeline."""
    from src.data.load_data import load_dataset
    from src.data.preprocess import preprocess_data
    from src.data.split_data import split_and_scale
    from src.models.train_model import train_model, save_model

    logger.info("--- Starting Training Pipeline ---")
    
    # 1. Load
    df = load_dataset()
    
    # 2. Preprocess
    df_processed = preprocess_data(df)
    
    # 3. Split & Scale
    feature_columns = df_processed.drop("stress_level", axis=1).columns.tolist()
    X_train, X_test, y_train, y_test = split_and_scale(df_processed)
    
    # 4. Train
    model = train_model(X_train, y_train)
    
    # 5. Save
    save_model(model, feature_columns)
    
    logger.info("--- Pipeline Completed Successfully ---")

def run_ui():
    """Launches the Streamlit UI."""
    ui_path = Path("src/ui/app_streamlit.py")
    if not ui_path.exists():
        logger.error(f"Streamlit UI file not found at {ui_path}")
        return

    logger.info("Launching Streamlit UI...")
    try:
        subprocess.run(["streamlit", "run", str(ui_path)], check=True)
    except KeyboardInterrupt:
        logger.info("Streamlit UI stopped by user.")
    except Exception as e:
        logger.error(f"Failed to launch Streamlit: {e}")

def main():
    parser = argparse.ArgumentParser(description="Minor Project Team 32 - Stress Analysis System")
    parser.add_argument("--train", action="store_true", help="Execute the training pipeline")
    parser.add_argument("--ui", action="store_true", help="Launch the Streamlit UI")
    
    args = parser.parse_args()

    if args.train:
        train_pipeline()
    elif args.ui:
        run_ui()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

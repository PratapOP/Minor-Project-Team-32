import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

def preprocess_data(df):
    """
    Cleans and encodes survey-based stress dataset
    """
    try:
        logger.info("Starting data preprocessing...")
        
        # 1. Remove duplicates
        df = df.drop_duplicates()

        # 2. Identify Target Column
        target_col = "Which type of stress do you primarily experience?"

        if target_col not in df.columns:
            logger.error("Target column not found")
            raise ValueError("Target column not found")

        # 3. Encode Target
        target_encoder = LabelEncoder()
        df["stress_level"] = target_encoder.fit_transform(df[target_col])

        # Save mapping for viva/demo
        base_path = Path(__file__).resolve().parent.parent.parent
        encoder_path = base_path / "models" / "target_encoder.pkl"
        joblib.dump(target_encoder, encoder_path)
        
        logger.info(f"Target Mapping: {dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))}")

        # Drop original target column
        df = df.drop(columns=[target_col])

        # 4. Handle Duplicate Columns
        df = df.loc[:, ~df.columns.duplicated()]

        # 5. Encode ALL categorical columns
        for col in df.columns:
            if df[col].dtype == "object":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        # 6. Final Check
        if "stress_level" not in df.columns:
            raise ValueError("Target encoding failed")

        logger.info("Preprocessing completed successfully.")
        return df
    except Exception as e:
        logger.exception("Error during preprocessing")
        raise

if __name__ == "__main__":
    from src.data.load_data import load_dataset
    try:
        df = load_dataset()
        df_processed = preprocess_data(df)
        print("\nPreprocessed Data Head:")
        print(df_processed.head())
    except Exception:
        pass
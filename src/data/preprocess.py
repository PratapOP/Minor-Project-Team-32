import pandas as pd
import logging
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

def preprocess_data(df):
    """
    Cleans and encodes survey-based stress dataset.
    """
    try:
        logger.info("Starting data preprocessing...")
        
        # 1. Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates()
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} duplicate rows.")

        # 2. Identify Target Column
        target_col = "Which type of stress do you primarily experience?"
        if target_col not in df.columns:
            logger.error(f"Target column '{target_col}' not found in dataset.")
            raise ValueError(f"Target column '{target_col}' not found. Available: {df.columns.tolist()}")

        # 3. Encode Target
        logger.info(f"Encoding target column: {target_col}")
        target_encoder = LabelEncoder()
        df["stress_level"] = target_encoder.fit_transform(df[target_col])
        
        # Save mapping for later use
        mapping = dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))
        logger.info(f"Target Mapping: {mapping}")
        
        # Save target encoder for inverse transform in UI
        base_path = Path(__file__).resolve().parent.parent.parent
        encoder_path = base_path / "models" / "target_encoder.pkl"
        joblib.dump(target_encoder, encoder_path)
        logger.info(f"Target encoder saved to {encoder_path}")

        # Drop original target column
        df = df.drop(columns=[target_col])

        # 4. Handle Duplicate Columns
        df = df.loc[:, ~df.columns.duplicated()]

        # 5. Encode Categorical Columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if not categorical_cols.empty:
            logger.info(f"Encoding categorical columns: {categorical_cols.tolist()}")
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                # Note: In a production app, you'd save encoders for each column.
                # For this project, we'll assume they are simple labels or don't change.

        # 6. Final verification
        if "stress_level" not in df.columns:
            raise ValueError("Target encoding failed - 'stress_level' missing.")

        logger.info("Preprocessing completed successfully.")
        return df

    except Exception as e:
        logger.exception("Error occurred during preprocessing")
        raise

if __name__ == "__main__":
    from src.data.load_data import load_dataset
    try:
        logging.basicConfig(level=logging.INFO)
        df = load_dataset()
        df_processed = preprocess_data(df)
        print("\nPreprocessed Data Head:")
        print(df_processed.head())
    except Exception:
        pass
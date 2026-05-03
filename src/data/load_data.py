import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset():
    """
    Loads the stress dataset from raw folder using robust path handling.
    """
    try:
        # Get project base directory
        base_path = Path(__file__).resolve().parent.parent.parent
        file_path = base_path / "data" / "raw" / "stress_dataset.csv"

        if not file_path.exists():
            logger.error(f"Dataset not found at: {file_path}")
            raise FileNotFoundError(f"Dataset not found at {file_path}")

        logger.info(f"Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset with shape: {df.shape}")

        return df
    except Exception as e:
        logger.exception("Error occurred while loading the dataset")
        raise

if __name__ == "__main__":
    try:
        df = load_dataset()
        print("\nDataset Preview:")
        print(df.head())
    except Exception:
        pass
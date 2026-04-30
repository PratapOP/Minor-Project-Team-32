import pandas as pd
import os


def load_dataset():
    """
    Loads the stress dataset from raw folder
    """

    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    file_path = os.path.join(base_path, "data", "raw", "stress_dataset.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError("Dataset not found at expected location")

    df = pd.read_csv(file_path)

    return df


if __name__ == "__main__":
    df = load_dataset()
    print("Dataset Loaded Successfully!")
    print(df.head())
    print("\nShape:", df.shape)
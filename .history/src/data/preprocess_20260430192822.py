import pandas as pd


def preprocess_data(df):
    """
    Cleans and validates dataset
    """

    # ---------------------------
    # 1. Remove Duplicates
    # ---------------------------
    df = df.drop_duplicates()

    # ---------------------------
    # 2. Handle Missing Values
    # ---------------------------
    if df.isnull().sum().sum() > 0:
        df = df.fillna(df.median(numeric_only=True))

    # ---------------------------
    # 3. Validate Ranges (IMPORTANT)
    # ---------------------------
    # These ranges are based on dataset understanding
    columns_range = {
        "anxiety_level": (0, 100),
        "self_esteem": (0, 100),
        "mental_health_history": (0, 1),
        "depression": (0, 100),
        "headache": (0, 5),
        "blood_pressure": (0, 5),
        "sleep_quality": (0, 5),
        "breathing_problem": (0, 5),
        "noise_level": (0, 5),
        "living_conditions": (0, 5),
        "safety": (0, 5),
        "basic_needs": (0, 5),
        "academic_performance": (0, 5),
        "study_load": (0, 5),
        "teacher_student_relationship": (0, 5),
        "future_career_concerns": (0, 5),
        "social_support": (0, 5),
        "peer_pressure": (0, 5),
        "extracurricular_activities": (0, 5),
        "bullying": (0, 5),
    }

    for col, (min_val, max_val) in columns_range.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=min_val, upper=max_val)

    # ---------------------------
# 4. Standardize Target Column
# ---------------------------

possible_targets = ["stress_level", "stress", "Stress_Level", "stress level"]

found_target = None
for col in df.columns:
    if col.strip().lower() in [t.lower() for t in possible_targets]:
        found_target = col
        break

if found_target is None:
    raise ValueError("No valid target column found in dataset")

# Rename to standard name
df = df.rename(columns={found_target: "stress_level"})

    # ---------------------------
    # 5. Convert Data Types
    # ---------------------------
    df = df.apply(pd.to_numeric, errors="ignore")

    return df


if __name__ == "__main__":
    from load_data import load_dataset

    df = load_dataset()
    df_clean = preprocess_data(df)

    print("Preprocessing Completed!")
    print(df_clean.head())
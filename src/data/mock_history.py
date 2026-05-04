import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_history(days=30):
    """
    Generates plausible historical stress data for visualization.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Simulate a fluctuating stress level (0 to 100)
    # Adding some seasonality and randomness
    base_stress = 40 + 20 * np.sin(np.linspace(0, 4 * np.pi, days + 1))
    noise = np.random.normal(0, 5, days + 1)
    stress_levels = np.clip(base_stress + noise, 10, 95)
    
    # Identify peaks (stress > 80)
    peaks = [1 if x > 80 else 0 for x in stress_levels]
    
    df = pd.DataFrame({
        'Date': date_range,
        'Stress Level': stress_levels,
        'Is Peak': peaks
    })
    
    return df

def get_factor_correlations():
    """
    Mock correlation data for research heatmaps.
    """
    factors = ['Sleep', 'Workload', 'Physical', 'Social', 'Academic']
    data = np.array([
        [1.0, -0.6, 0.7, 0.4, -0.5],
        [-0.6, 1.0, -0.4, -0.3, 0.8],
        [0.7, -0.4, 1.0, 0.5, -0.3],
        [0.4, -0.3, 0.5, 1.0, -0.2],
        [-0.5, 0.8, -0.3, -0.2, 1.0]
    ])
    return pd.DataFrame(data, index=factors, columns=factors)

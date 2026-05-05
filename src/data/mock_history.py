import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_history_data(days=30):
    """
    Returns history data formatted for Chart.js
    """
    end_date = datetime.now()
    dates = [(end_date - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)][::-1]
    
    base_stress = 40 + 20 * np.sin(np.linspace(0, 4 * np.pi, days))
    noise = np.random.normal(0, 5, days)
    stress_levels = np.clip(base_stress + noise, 10, 95).tolist()
    
    return {
        'dates': dates,
        'levels': [round(x, 1) for x in stress_levels],
        'peaks': [1 if x > 80 else 0 for x in stress_levels]
    }

def get_research_data():
    """
    Returns correlation and performance metrics for the Research Lab
    """
    factors = ['Sleep', 'Workload', 'Physical', 'Social', 'Academic']
    corr_matrix = np.array([
        [1.0, -0.6, 0.7, 0.4, -0.5],
        [-0.6, 1.0, -0.4, -0.3, 0.8],
        [0.7, -0.4, 1.0, 0.5, -0.3],
        [0.4, -0.3, 0.5, 1.0, -0.2],
        [-0.5, 0.8, -0.3, -0.2, 1.0]
    ])
    
    return {
        'history': get_history_data(),
        'correlations': {
            'index': factors,
            'columns': factors,
            'values': corr_matrix.tolist()
        }
    }

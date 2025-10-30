"""
Configuration file for ML Pipeline
"""

import os

# Data directories
DATA_DIRS = {
    'SHORT': '/data/short_soaking_time_10072025/Tc_keller',
    'STANDARD': '/data/standard_soaking_time_10072025/tc_keller',
    'LONG': '/data/long_soaking_time_10072025/tc_keller'
}

# Temperature limits
MIN_TEMP = 500
MAX_TEMP = 830

# Output directories
OUTPUT_DIR = 'ml_output_2'
DATA_PKL_DIR = os.path.join(OUTPUT_DIR, 'data_pkl')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')

# Create directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_PKL_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# PKL file paths
RAW_DATA_PKL = os.path.join(DATA_PKL_DIR, 'raw_data.pkl')
FEATURED_DATA_PKL = os.path.join(DATA_PKL_DIR, 'featured_data.pkl')
TRAIN_TEST_PKL = os.path.join(DATA_PKL_DIR, 'train_test_splits.pkl')
MODELS_PKL = os.path.join(DATA_PKL_DIR, 'all_models.pkl')
BEST_MODEL_PKL = os.path.join(MODEL_DIR, 'best_model.pkl')

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Soaking types and thicknesses
SOAKING_TYPES = ['SHORT', 'STANDARD', 'LONG']
THICKNESSES = [10, 15, 20]

# Colors for plots
COLORS = {
    'primary': '#0173B2',
    'secondary': '#DE8F05',
    'tertiary': '#029E73',
    'error': '#CC78BC',
    'line': '#D55E00',
    'gray': '#949494',
    'dark': '#333333',
}
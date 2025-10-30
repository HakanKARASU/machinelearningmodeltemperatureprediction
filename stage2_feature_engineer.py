"""
STAGE 2: Feature Engineering
Load RAW_DATA.pkl → Add features → Save FEATURED_DATA.pkl
"""

import pickle
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from config import *

def add_features(df):
    """Add engineered features"""
    print("\n" + "="*70)
    print("STAGE 2: FEATURE ENGINEERING")
    print("="*70)
    
    df = df.copy()
    
    # Time features
    df['time_seconds'] = df['Time']
    df['time_minutes'] = df['Time'] / 60
    df['time_squared'] = df['time_seconds'] ** 2
    
    # Smoothed signals
    df['Impac_smooth'] = gaussian_filter1d(df['Impac'], sigma=5)
    df['Keller_smooth'] = gaussian_filter1d(df['Keller'], sigma=5)
    
    # Derivatives
    df['Impac_rate'] = np.gradient(df['Impac_smooth'], df['time_seconds'])
    df['Keller_rate'] = np.gradient(df['Keller_smooth'], df['time_seconds'])
    df['Impac_accel'] = np.gradient(df['Impac_rate'], df['time_seconds'])
    df['Keller_accel'] = np.gradient(df['Keller_rate'], df['time_seconds'])
    
    # Rolling statistics
    for col in ['Impac', 'Keller']:
        for window in [5, 10, 20]:
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window, center=True).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window, center=True).std()
    
    # Sensor relationships
    df['Impac_Keller_ratio'] = df['Impac'] / (df['Keller'] + 1e-10)
    df['Impac_Keller_diff'] = df['Impac'] - df['Keller']
    df['Pyrometer_avg'] = (df['Impac'] + df['Keller']) / 2
    df['Rate_ratio'] = df['Impac_rate'] / (df['Keller_rate'] + 1e-10)
    df['Rate_diff'] = df['Impac_rate'] - df['Keller_rate']
    df['Rate_avg'] = (df['Impac_rate'] + df['Keller_rate']) / 2
    
    # Drop NaN
    df = df.dropna().reset_index(drop=True)
    
    print(f"✅ Features added: {df.shape[1]} columns")
    print(f"   Samples: {len(df):,}")
    
    # Save to PKL
    with open(FEATURED_DATA_PKL, 'wb') as f:
        pickle.dump(df, f)
    print(f"\n💾 Saved: {FEATURED_DATA_PKL}")
    
    return df


if __name__ == "__main__":
    # Load raw data
    with open(RAW_DATA_PKL, 'rb') as f:
        df = pickle.load(f)
    
    # Add features
    add_features(df)
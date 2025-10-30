"""
STAGE 3: Data Splitting with Stratification
Load FEATURED_DATA.pkl → Stratified split → Save TRAIN_TEST.pkl
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from config import *

def remove_outliers(df):
    """Remove worst 2 outliers based on residuals"""
    print("\n🧹 Removing outliers...")
    
    exclude = ['X_Value', 'TC', 'Time', 'Impac_smooth', 'Keller_smooth', 
               'soaking_label', 'filename']
    features = [col for col in df.columns if col not in exclude]
    
    X = df[features].values
    y = df['TC'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_scaled, y)
    
    y_pred = model.predict(X_scaled)
    residuals = np.abs(y - y_pred)
    
    worst_2_indices = np.argsort(residuals)[-2:][::-1]
    
    print(f"  Removing 2 outliers with residuals: {residuals[worst_2_indices[0]]:.2f}°C, {residuals[worst_2_indices[1]]:.2f}°C")
    
    df_clean = df.drop(df.index[worst_2_indices]).reset_index(drop=True)
    print(f"  ✅ Cleaned: {len(df_clean):,} samples")
    
    return df_clean, features


def stratified_split(df, features):
    """
    Stratified splitting based on:
    - Soaking time (SHORT, STANDARD, LONG)
    - Thickness (10mm, 15mm, 20mm)
    """
    print("\n" + "="*70)
    print("STAGE 3: STRATIFIED DATA SPLITTING")
    print("="*70)
    
    # Create stratification key
    df['strat_key'] = df['soaking_label'] + '_' + df['thickness_mm'].astype(str) + 'mm'
    
    print(f"\nStratification groups:")
    print(df['strat_key'].value_counts().sort_index())
    
    X = df[features].values
    y = df['TC'].values
    
    # Stratified split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=df['strat_key']
    )
    
    # Get metadata for train/test
    train_meta = df.loc[idx_train, ['soaking_label', 'thickness_mm', 'strat_key']].reset_index(drop=True)
    test_meta = df.loc[idx_test, ['soaking_label', 'thickness_mm', 'strat_key']].reset_index(drop=True)
    
    print(f"\n✅ Split complete:")
    print(f"   Training: {len(X_train):,} samples")
    print(f"   Testing: {len(X_test):,} samples")
    
    print(f"\n📊 Training set distribution:")
    print(train_meta['strat_key'].value_counts().sort_index())
    
    print(f"\n📊 Testing set distribution:")
    print(test_meta['strat_key'].value_counts().sort_index())
    
    # Save to PKL
    split_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'train_meta': train_meta,
        'test_meta': test_meta,
        'features': features,
        'df_clean': df
    }
    
    with open(TRAIN_TEST_PKL, 'wb') as f:
        pickle.dump(split_data, f)
    print(f"\n💾 Saved: {TRAIN_TEST_PKL}")
    
    return split_data


if __name__ == "__main__":
    # Load featured data
    with open(FEATURED_DATA_PKL, 'rb') as f:
        df = pickle.load(f)
    
    # Remove outliers
    df_clean, features = remove_outliers(df)
    
    # Stratified split
    stratified_split(df_clean, features)

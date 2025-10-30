"""
STAGE 7: Subset Analysis
Train models on different subsets:
- By soaking time (SHORT, STANDARD, LONG)
- By thickness (10mm, 15mm, 20mm)
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from config import *

MODEL_NAMES = {
    'random_forest': 'Random Forest',
    'gradient_boost': 'Gradient Boosting',
    'neural_net': 'Neural Network'
}

def train_on_subset(df, features, best_model_name, subset_filter, subset_value, label):
    """Train model on a specific subset"""
    
    # Filter data
    if subset_filter == 'soaking':
        subset_df = df[df['soaking_label'] == subset_value].copy()
    else:  # thickness
        subset_df = df[df['thickness_mm'] == subset_value].copy()
    
    if len(subset_df) < 20:
        print(f"  ⚠️  {label}: Too few samples ({len(subset_df)})")
        return None
    
    # Prepare data
    X = subset_df[features].values
    y = subset_df['TC'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Select model type
    if best_model_name == 'random_forest':
        model = RandomForestRegressor(n_estimators=200, max_depth=20, 
                                     random_state=RANDOM_STATE, n_jobs=-1)
    elif best_model_name == 'gradient_boost':
        model = GradientBoostingRegressor(n_estimators=200, max_depth=8, 
                                         random_state=RANDOM_STATE)
    else:
        model = MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=500, 
                            random_state=RANDOM_STATE)
    
    # Train
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"  ✅ {label}: RMSE={rmse:.2f}°C, R²={r2:.4f}, n={len(y_test)}")
    
    return {
        'model': model,
        'scaler': scaler,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'y_test': y_test,
        'y_pred': y_pred
    }


def analyze_subsets():
    """Analyze all subsets"""
    print("\n" + "="*70)
    print("STAGE 7: SUBSET ANALYSIS")
    print("="*70)
    
    # Load data
    with open(TRAIN_TEST_PKL, 'rb') as f:
        split_data = pickle.load(f)
    
    with open(BEST_MODEL_PKL, 'rb') as f:
        best_data = pickle.load(f)
    
    df = split_data['df_clean']
    features = split_data['features']
    best_model_name = best_data['model_name']
    
    print(f"\n🏆 Using: {MODEL_NAMES[best_model_name]}")
    
    # Analyze by soaking time
    print("\n" + "-"*70)
    print("TRAINING BY SOAKING TIME")
    print("-"*70)
    
    soaking_results = {}
    for soaking in SOAKING_TYPES:
        result = train_on_subset(df, features, best_model_name, 
                                'soaking', soaking, soaking)
        if result:
            soaking_results[soaking] = result
    
    # Analyze by thickness
    print("\n" + "-"*70)
    print("TRAINING BY THICKNESS")
    print("-"*70)
    
    thickness_results = {}
    for thickness in THICKNESSES:
        label = f"{thickness}mm"
        result = train_on_subset(df, features, best_model_name, 
                                'thickness', thickness, label)
        if result:
            thickness_results[label] = result
    
    # Save results
    subset_data = {
        'soaking_results': soaking_results,
        'thickness_results': thickness_results,
        'best_model_name': best_model_name
    }
    
    subset_pkl = f'{DATA_PKL_DIR}/subset_results.pkl'
    with open(subset_pkl, 'wb') as f:
        pickle.dump(subset_data, f)
    print(f"\n💾 Saved: {subset_pkl}")
    
    return subset_data


if __name__ == "__main__":
    analyze_subsets()

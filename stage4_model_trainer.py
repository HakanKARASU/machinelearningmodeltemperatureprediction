"""
STAGE 4: Model Training
Load TRAIN_TEST.pkl → Train 3 models → Save MODELS.pkl
"""

import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from config import *

MODEL_NAMES = {
    'random_forest': 'Random Forest',
    'gradient_boost': 'Gradient Boosting',
    'neural_net': 'Neural Network'
}

def train_all_models(X_train, y_train, X_test, y_test):
    """Train all 3 models"""
    print("\n" + "="*70)
    print("STAGE 4: MODEL TRAINING")
    print("="*70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=200, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'gradient_boost': GradientBoostingRegressor(
            n_estimators=200, max_depth=8, random_state=RANDOM_STATE
        ),
        'neural_net': MLPRegressor(
            hidden_layer_sizes=(100, 50, 25), max_iter=500, random_state=RANDOM_STATE
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n🔄 Training {MODEL_NAMES[name]}...")
        
        model.fit(X_train_scaled, y_train)
        
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, 
            cv=CV_FOLDS, scoring='neg_mean_squared_error', n_jobs=-1
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        cv_std = np.sqrt(cv_scores.std())
        
        results[name] = {
            'model': model,
            'scaler': scaler,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'mae': test_mae,
            'r2': test_r2,
            'cv_rmse': cv_rmse,
            'cv_std': cv_std,
            'n_train': len(y_train),
            'n_test': len(y_test)
        }
        
        print(f"  ✅ Train RMSE: {train_rmse:.2f}°C")
        print(f"  ✅ Test RMSE:  {test_rmse:.2f}°C")
        print(f"  ✅ Test R²:    {test_r2:.4f}")
        print(f"  ✅ CV RMSE:    {cv_rmse:.2f} ± {cv_std:.2f}°C")
    
    # Save all models
    with open(MODELS_PKL, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n💾 Saved all models: {MODELS_PKL}")
    
    return results


if __name__ == "__main__":
    # Load train/test split
    with open(TRAIN_TEST_PKL, 'rb') as f:
        split_data = pickle.load(f)
    
    # Train models
    train_all_models(
        split_data['X_train'],
        split_data['y_train'],
        split_data['X_test'],
        split_data['y_test']
    )
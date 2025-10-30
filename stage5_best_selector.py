"""
STAGE 5: Best Model Selection
Load MODELS.pkl → Select best → Save BEST_MODEL.pkl
"""

import pickle
from config import *

MODEL_NAMES = {
    'random_forest': 'Random Forest',
    'gradient_boost': 'Gradient Boosting',
    'neural_net': 'Neural Network'
}

def select_best_model():
    """Select best model based on test RMSE"""
    print("\n" + "="*70)
    print("STAGE 5: BEST MODEL SELECTION")
    print("="*70)
    
    # Load all models
    with open(MODELS_PKL, 'rb') as f:
        results = pickle.load(f)
    
    # Find best model
    best_name = min(results.items(), key=lambda x: x[1]['test_rmse'])[0]
    best_result = results[best_name]
    
    print(f"\n🏆 BEST MODEL: {MODEL_NAMES[best_name]}")
    print(f"   Test RMSE: {best_result['test_rmse']:.2f}°C")
    print(f"   Test MAE:  {best_result['mae']:.2f}°C")
    print(f"   Test R²:   {best_result['r2']:.4f}")
    print(f"   CV RMSE:   {best_result['cv_rmse']:.2f} ± {best_result['cv_std']:.2f}°C")
    
    # Load features
    with open(TRAIN_TEST_PKL, 'rb') as f:
        split_data = pickle.load(f)
    
    # Save best model
    best_model_data = {
        'model_name': best_name,
        'model': best_result['model'],
        'scaler': best_result['scaler'],
        'features': split_data['features'],
        'metrics': {
            'test_rmse': best_result['test_rmse'],
            'test_mae': best_result['mae'],
            'test_r2': best_result['r2'],
            'cv_rmse': best_result['cv_rmse'],
            'cv_std': best_result['cv_std']
        }
    }
    
    with open(BEST_MODEL_PKL, 'wb') as f:
        pickle.dump(best_model_data, f)
    print(f"\n💾 Saved best model: {BEST_MODEL_PKL}")
    
    return best_name, results


if __name__ == "__main__":
    select_best_model()

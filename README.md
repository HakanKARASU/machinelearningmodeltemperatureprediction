# ML Temperature Prediction Pipeline

Machine learning pipeline for predicting thermocouple temperatures from pyrometer data using stratified validation.

## 🏆 Results

- **Best Model:** Gradient Boosting
- **RMSE:** 2.03°C
- **R²:** 0.9995 (99.95% accuracy)
- **Test Samples:** 3,661

## 🚀 Quick Start with Docker
```bash
# Build Docker image
docker build -t ml-pipeline:latest .

# Run pipeline
docker run \
  -v /path/to/your/data:/data:ro \
  -v $(pwd)/ml_output_2:/app/ml_output_2 \
  ml-pipeline:latest
```

## 📊 Pipeline Stages

1. **Data Loading** → `raw_data.pkl`
2. **Feature Engineering** → `featured_data.pkl`
3. **Stratified Splitting** → `train_test_splits.pkl`
4. **Model Training** → `all_models.pkl`
5. **Best Model Selection** → `best_model.pkl`
6. **Visualizations** → Advanced parity plots
7. **Subset Analysis** → By soaking time & thickness
8. **Comparison Plots** → Comprehensive analysis

## 🔬 Model Performance

### Overall
| Model | RMSE (°C) | R² |
|-------|-----------|-----|
| **Gradient Boosting** | **2.03** | **0.9995** |
| Random Forest | 3.13 | 0.9988 |
| Neural Network | 4.11 | 0.9979 |

## 📁 Project Structure
```
├── config.py                    # Configuration
├── stage1_data_loader.py        # Load LVM files
├── stage2_feature_engineer.py   # Feature engineering
├── stage3_data_splitter.py      # Stratified splitting
├── stage4_model_trainer.py      # Train models
├── stage5_best_selector.py      # Select best model
├── stage6_visualizer.py         # Visualizations
├── stage7_subset_analyzer.py    # Subset analysis
├── stage8_comparator.py         # Comparison plots
├── main.py                      # Orchestrator
└── Dockerfile                   # Docker config
```

## 📄 License

MIT

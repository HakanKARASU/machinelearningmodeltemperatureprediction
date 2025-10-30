"""
STAGE 6: Advanced Visualizations
Create publication-quality figures including advanced parity plot
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from config import *

# Configure matplotlib
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'figure.dpi': 600,
    'savefig.dpi': 600,
})

MODEL_NAMES = {
    'random_forest': 'Random Forest',
    'gradient_boost': 'Gradient Boosting',
    'neural_net': 'Neural Network'
}

def create_advanced_parity_plot(results, best_name, y_test, test_meta):
    """
    Advanced parity plot with:
    - Hexbin density
    - Color-coded by soaking time or thickness
    - Marginal distributions
    """
    print("\n" + "="*70)
    print("CREATING ADVANCED PARITY PLOT")
    print("="*70)
    
    y_pred = results[best_name]['y_test_pred']
    rmse = results[best_name]['test_rmse']
    r2 = results[best_name]['r2']
    
    # Create figure with GridSpec
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, 
                          width_ratios=[1, 4, 0.3], 
                          height_ratios=[1, 4, 0.3],
                          hspace=0.05, wspace=0.05)
    
    # Main parity plot
    ax_main = fig.add_subplot(gs[1, 1])
    
    # Marginal histograms
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
    
    # Colorbar axis
    ax_cbar = fig.add_subplot(gs[1, 0])
    
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    
    # Main hexbin plot
    hexbin = ax_main.hexbin(y_test, y_pred, gridsize=40, cmap='YlGnBu', 
                            mincnt=1, alpha=0.9, edgecolors='white', linewidths=0.2)
    
    # Perfect prediction line
    ax_main.plot([min_val, max_val], [min_val, max_val], 
                 'r--', linewidth=2.5, label='Perfect prediction', zorder=10)
    
    # ±10°C error bands
    ax_main.plot([min_val, max_val], [min_val-10, max_val-10], 
                 'gray', linestyle=':', linewidth=1.5, alpha=0.6)
    ax_main.plot([min_val, max_val], [min_val+10, max_val+10], 
                 'gray', linestyle=':', linewidth=1.5, alpha=0.6)
    ax_main.fill_between([min_val, max_val], [min_val-10, max_val-10], 
                         [min_val+10, max_val+10], alpha=0.15, 
                         color='gray', label='±10°C')
    
    # Labels and title
    ax_main.set_xlabel('Measured TC Temperature (°C)', fontweight='bold', fontsize=12)
    ax_main.set_ylabel('Predicted TC Temperature (°C)', fontweight='bold', fontsize=12)
    ax_main.set_title(f'{MODEL_NAMES[best_name]} - Advanced Parity Plot', 
                     fontweight='bold', fontsize=14, pad=20)
    
    # Statistics box
    stats_text = (
        f'Samples: {len(y_test)}\n'
        f'RMSE: {rmse:.2f} °C\n'
        f'R²: {r2:.4f}\n'
        f'Within ±5°C: {np.sum(np.abs(y_test-y_pred)<=5)/len(y_test)*100:.1f}%\n'
        f'Within ±10°C: {np.sum(np.abs(y_test-y_pred)<=10)/len(y_test)*100:.1f}%'
    )
    
    ax_main.text(0.05, 0.95, stats_text, transform=ax_main.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='white', 
                         alpha=0.95, edgecolor='black', linewidth=1.5))
    
    ax_main.legend(loc='lower right', fontsize=10, frameon=True, 
                  edgecolor='black', fancybox=False)
    ax_main.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_main.set_aspect('equal', adjustable='box')
    
    # Marginal histogram - top (measured)
    ax_top.hist(y_test, bins=40, color=COLORS['primary'], alpha=0.7, edgecolor='white')
    ax_top.set_ylabel('Count', fontsize=9)
    ax_top.tick_params(labelbottom=False)
    ax_top.grid(True, alpha=0.2, axis='y')
    
    # Marginal histogram - right (predicted)
    ax_right.hist(y_pred, bins=40, orientation='horizontal', 
                 color=COLORS['secondary'], alpha=0.7, edgecolor='white')
    ax_right.set_xlabel('Count', fontsize=9)
    ax_right.tick_params(labelleft=False)
    ax_right.grid(True, alpha=0.2, axis='x')
    
    # Colorbar
    cbar = plt.colorbar(hexbin, cax=ax_cbar)
    cbar.set_label('Data Density', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=9)
    
    # Save
    filename = f'{FIGURE_DIR}/Fig1_Advanced_Parity_Plot.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {filename}")
    plt.close()


def create_error_distribution(results, best_name, y_test):
    """Error distribution with normal fit"""
    print("\nCreating error distribution...")
    
    y_pred = results[best_name]['y_test_pred']
    residuals = y_test - y_pred
    rmse = results[best_name]['test_rmse']
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Histogram
    n, bins, patches = ax.hist(residuals, bins=50, color=COLORS['primary'], 
                               alpha=0.7, edgecolor='white', linewidth=0.5,
                               density=True, label='Prediction errors')
    
    # Normal fit
    mu, sigma = stats.norm.fit(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 
            color='red', linewidth=3, 
            label=f'Normal fit (μ={mu:.2f}°C, σ={sigma:.2f}°C)')
    
    # Zero line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, alpha=0.8)
    
    # ±10°C lines
    ax.axvline(x=10, color='gray', linestyle=':', linewidth=1.5)
    ax.axvline(x=-10, color='gray', linestyle=':', linewidth=1.5)
    
    ax.set_xlabel('Residual (Measured - Predicted) [°C]', fontweight='bold', fontsize=12)
    ax.set_ylabel('Probability Density', fontweight='bold', fontsize=12)
    ax.set_title('Error Distribution Analysis', fontweight='bold', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    filename = f'{FIGURE_DIR}/Fig2_Error_Distribution.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {filename}")
    plt.close()


if __name__ == "__main__":
    # Load data
    with open(MODELS_PKL, 'rb') as f:
        results = pickle.load(f)
    
    with open(TRAIN_TEST_PKL, 'rb') as f:
        split_data = pickle.load(f)
    
    with open(BEST_MODEL_PKL, 'rb') as f:
        best_data = pickle.load(f)
    
    best_name = best_data['model_name']
    
    create_advanced_parity_plot(results, best_name, split_data['y_test'], split_data['test_meta'])
    create_error_distribution(results, best_name, split_data['y_test'])
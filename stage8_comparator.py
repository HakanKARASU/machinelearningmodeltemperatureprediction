"""
STAGE 8: Comparison Visualizations
Create comparison plots for different subsets
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from config import *

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

def create_stratified_split_visualization():
    """
    Create visualization similar to your paper's Figure S4
    Showing stratified splitting method
    """
    print("\n" + "="*70)
    print("CREATING STRATIFIED SPLIT VISUALIZATION")
    print("="*70)
    
    # Load data
    with open(TRAIN_TEST_PKL, 'rb') as f:
        split_data = pickle.load(f)
    
    train_meta = split_data['train_meta']
    test_meta = split_data['test_meta']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Stratified Data Splitting Method', fontsize=16, fontweight='bold')
    
    # Plot for each soaking time
    for idx, soaking in enumerate(SOAKING_TYPES):
        ax = axes[0, idx]
        
        # Filter data
        train_subset = train_meta[train_meta['soaking_label'] == soaking]
        test_subset = test_meta[test_meta['soaking_label'] == soaking]
        
        # Count by thickness
        train_counts = train_subset['thickness_mm'].value_counts().sort_index()
        test_counts = test_subset['thickness_mm'].value_counts().sort_index()
        
        # Create pie chart
        labels = [f"{t}mm" for t in train_counts.index]
        sizes_train = train_counts.values
        sizes_test = test_counts.values
        
        colors_train = ['#4472C4', '#ED7D31', '#A5A5A5']
        colors_test = ['#70AD47', '#FFC000', '#5B9BD5']
        
        # Combined data
        wedges, texts, autotexts = ax.pie(
            list(sizes_train) + list(sizes_test),
            labels=labels + labels,
            colors=colors_train + colors_test,
            autopct='%1.0f%%',
            startangle=90
        )
        
        ax.set_title(f'{soaking} Soaking Time', fontweight='bold', fontsize=12)
    
    # Plot for each thickness
    for idx, thickness in enumerate(THICKNESSES):
        ax = axes[1, idx]
        
        # Filter data
        train_subset = train_meta[train_meta['thickness_mm'] == thickness]
        test_subset = test_meta[test_meta['thickness_mm'] == thickness]
        
        # Count by soaking
        train_counts = train_subset['soaking_label'].value_counts()
        test_counts = test_subset['soaking_label'].value_counts()
        
        # Create pie chart
        labels = train_counts.index.tolist()
        sizes_train = train_counts.values
        sizes_test = test_counts.values
        
        colors = ['#4472C4', '#ED7D31', '#A5A5A5', '#70AD47', '#FFC000', '#5B9BD5']
        
        ax.pie(
            list(sizes_train) + list(sizes_test),
            labels=labels + labels,
            colors=colors,
            autopct='%1.0f%%',
            startangle=90
        )
        
        ax.set_title(f'{thickness}mm Thickness', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    filename = f'{FIGURE_DIR}/Fig_Stratified_Splitting.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {filename}")
    plt.close()


def create_subset_comparison_plot():
    """Create comparison bar plot for all subsets"""
    print("\nCreating subset comparison plot...")
    
    # Load data
    with open(f'{DATA_PKL_DIR}/subset_results.pkl', 'rb') as f:
        subset_data = pickle.load(f)
    
    with open(BEST_MODEL_PKL, 'rb') as f:
        best_data = pickle.load(f)
    
    soaking_results = subset_data['soaking_results']
    thickness_results = subset_data['thickness_results']
    best_metrics = best_data['metrics']
    best_name = best_data['model_name']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Soaking time comparison
    ax1 = axes[0]
    soaking_labels = list(soaking_results.keys())
    soaking_rmse = [soaking_results[k]['rmse'] for k in soaking_labels]
    soaking_r2 = [soaking_results[k]['r2'] for k in soaking_labels]
    
    x = np.arange(len(soaking_labels))
    width = 0.35
    
    ax1.bar(x - width/2, soaking_rmse, width, label='RMSE (°C)', 
            color=COLORS['primary'], alpha=0.8)
    ax1_twin = ax1.twinx()
    ax1_twin.bar(x + width/2, soaking_r2, width, label='R²', 
                 color=COLORS['secondary'], alpha=0.8)
    
    ax1.axhline(y=best_metrics['test_rmse'], color='red', linestyle='--', 
                linewidth=2, label=f'Overall RMSE: {best_metrics["test_rmse"]:.2f}°C')
    
    ax1.set_xlabel('Soaking Time', fontweight='bold', fontsize=12)
    ax1.set_ylabel('RMSE (°C)', fontweight='bold', fontsize=12, color=COLORS['primary'])
    ax1_twin.set_ylabel('R²', fontweight='bold', fontsize=12, color=COLORS['secondary'])
    ax1.set_title('Performance by Soaking Time', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(soaking_labels, fontsize=11)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Thickness comparison
    ax2 = axes[1]
    thickness_labels = list(thickness_results.keys())
    thickness_rmse = [thickness_results[k]['rmse'] for k in thickness_labels]
    thickness_r2 = [thickness_results[k]['r2'] for k in thickness_labels]
    
    x = np.arange(len(thickness_labels))
    
    ax2.bar(x - width/2, thickness_rmse, width, label='RMSE (°C)', 
            color=COLORS['tertiary'], alpha=0.8)
    ax2_twin = ax2.twinx()
    ax2_twin.bar(x + width/2, thickness_r2, width, label='R²', 
                 color=COLORS['error'], alpha=0.8)
    
    ax2.axhline(y=best_metrics['test_rmse'], color='red', linestyle='--', 
                linewidth=2, label=f'Overall RMSE: {best_metrics["test_rmse"]:.2f}°C')
    
    ax2.set_xlabel('Thickness', fontweight='bold', fontsize=12)
    ax2.set_ylabel('RMSE (°C)', fontweight='bold', fontsize=12, color=COLORS['tertiary'])
    ax2_twin.set_ylabel('R²', fontweight='bold', fontsize=12, color=COLORS['error'])
    ax2.set_title('Performance by Thickness', fontweight='bold', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(thickness_labels, fontsize=11)
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    filename = f'{FIGURE_DIR}/Fig_Subset_Comparison.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {filename}")
    plt.close()


def create_comprehensive_table():
    """Create comprehensive comparison table"""
    print("\nCreating comprehensive comparison table...")
    
    # Load all data
    with open(f'{DATA_PKL_DIR}/subset_results.pkl', 'rb') as f:
        subset_data = pickle.load(f)
    
    with open(BEST_MODEL_PKL, 'rb') as f:
        best_data = pickle.load(f)
    
    with open(MODELS_PKL, 'rb') as f:
        all_models = pickle.load(f)
    
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = [['Model/Subset', 'RMSE (°C)', 'MAE (°C)', 'R²', 'Train Samples', 'Test Samples']]
    
    # All models comparison
    table_data.append(['═'*40, '═'*12, '═'*12, '═'*10, '═'*14, '═'*14])
    table_data.append(['ALL MODELS COMPARISON', '', '', '', '', ''])
    table_data.append(['─'*40, '─'*12, '─'*12, '─'*10, '─'*14, '─'*14])
    
    for model_name in ['random_forest', 'gradient_boost', 'neural_net']:
        r = all_models[model_name]
        table_data.append([
            MODEL_NAMES[model_name],
            f"{r['test_rmse']:.2f}",
            f"{r['mae']:.2f}",
            f"{r['r2']:.4f}",
            f"{r['n_train']}",
            f"{r['n_test']}"
        ])
    
    # Best model overall
    best_name = best_data['model_name']
    best_metrics = best_data['metrics']
    table_data.append(['═'*40, '═'*12, '═'*12, '═'*10, '═'*14, '═'*14])
    table_data.append([
        f"BEST: {MODEL_NAMES[best_name]}",
        f"{best_metrics['test_rmse']:.2f}",
        '-',
        f"{best_metrics['test_r2']:.4f}",
        '-',
        '-'
    ])
    
    # Soaking time subsets
    table_data.append(['═'*40, '═'*12, '═'*12, '═'*10, '═'*14, '═'*14])
    table_data.append(['BY SOAKING TIME', '', '', '', '', ''])
    table_data.append(['─'*40, '─'*12, '─'*12, '─'*10, '─'*14, '─'*14])
    
    soaking_results = subset_data['soaking_results']
    for label, r in soaking_results.items():
        table_data.append([
            f"  {label}",
            f"{r['rmse']:.2f}",
            f"{r['mae']:.2f}",
            f"{r['r2']:.4f}",
            f"{r['n_train']}",
            f"{r['n_test']}"
        ])
    
    # Thickness subsets
    table_data.append(['═'*40, '═'*12, '═'*12, '═'*10, '═'*14, '═'*14])
    table_data.append(['BY THICKNESS', '', '', '', '', ''])
    table_data.append(['─'*40, '─'*12, '─'*12, '─'*10, '─'*14, '─'*14])
    
    thickness_results = subset_data['thickness_results']
    for label, r in thickness_results.items():
        table_data.append([
            f"  {label}",
            f"{r['rmse']:.2f}",
            f"{r['mae']:.2f}",
            f"{r['r2']:.4f}",
            f"{r['n_train']}",
            f"{r['n_test']}"
        ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.30, 0.14, 0.14, 0.12, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(6):
        table[(0, i)].set_facecolor('#2c3e50')
        table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)
    
    # Highlight best model
    best_row = 5  # Adjust based on table structure
    for i in range(6):
        table[(best_row, i)].set_facecolor('#27ae60')
        table[(best_row, i)].set_text_props(weight='bold', color='white')
    
    plt.title(f'Comprehensive Model Performance Analysis\nBest Model: {MODEL_NAMES[best_name]} (RMSE: {best_metrics["test_rmse"]:.2f}°C)',
             fontsize=14, fontweight='bold', pad=20)
    
    filename = f'{FIGURE_DIR}/Table_Comprehensive_Comparison.png'
    plt.savefig(filename, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved: {filename}")
    plt.close()


if __name__ == "__main__":
    create_stratified_split_visualization()
    create_subset_comparison_plot()
    create_comprehensive_table()

"""
MAIN PIPELINE ORCHESTRATOR
Runs all stages in sequence with checkpoint support
"""

import os
import sys
import subprocess

def run_pipeline(start_stage=1):
    """
    Run ML pipeline from specified stage
    
    Stages:
    1. Data Loading
    2. Feature Engineering
    3. Data Splitting (Stratified)
    4. Model Training
    5. Best Model Selection
    6. Visualizations
    7. Subset Analysis
    8. Comparison Plots
    """
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  ML PIPELINE - MODULAR WITH CHECKPOINTS".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    stages = {
        1: ("Data Loading", "stage1_data_loader.py"),
        2: ("Feature Engineering", "stage2_feature_engineer.py"),
        3: ("Data Splitting", "stage3_data_splitter.py"),
        4: ("Model Training", "stage4_model_trainer.py"),
        5: ("Best Model Selection", "stage5_best_selector.py"),
        6: ("Visualizations", "stage6_visualizer.py"),
        7: ("Subset Analysis", "stage7_subset_analyzer.py"),
        8: ("Comparison Plots", "stage8_comparator.py"),
    }
    
    for stage_num in range(start_stage, 9):
        stage_name, script_name = stages[stage_num]
        
        print(f"\n{'='*70}")
        print(f"  STAGE {stage_num}: {stage_name}")
        print(f"{'='*70}\n")
        
        try:
            # Run the stage script
            result = subprocess.run(
                ['python3', script_name],
                check=True,
                capture_output=False
            )
            
            print(f"\n✅ Stage {stage_num} complete!")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Stage {stage_num} failed!")
            print(f"   You can restart from stage {stage_num} using:")
            print(f"   docker run ... ml-pipeline:latest python3 main.py {stage_num}")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Stage {stage_num} error: {e}")
            sys.exit(1)
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  PIPELINE COMPLETE!".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    print("📊 Generated Outputs:")
    print(f"   Data PKL files: ml_output_2/data_pkl/")
    print(f"   Figures: ml_output_2/figures/")
    print(f"   Models: ml_output_2/models/")


if __name__ == "__main__":
    # Check if starting stage specified
    start_stage = 1
    if len(sys.argv) > 1:
        start_stage = int(sys.argv[1])
        print(f"\n🔄 Resuming from Stage {start_stage}")
    
    run_pipeline(start_stage)
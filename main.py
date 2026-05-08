import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.feature_engineering import perform_feature_engineering
from modules.perception_ml import train_perception_models, predict_perception, FEATURES
from modules.knowledge_base import create_grid_map
from modules.rule_based import set_goal
from modules.path_planner import plan_path_with_fallback
from modules.visualizer import visualize_scenario, create_scenario_gif

GRID_WIDTH = 120
GRID_HEIGHT = 80

def main():
    print("=" * 60)
    print("  AUTONOMOUS DRIVING PIPELINE")
    print("=" * 60)
    
    # 1. Load Data
    print("\n1. Loading Data...")
    dataset_path = 'data/autonomous_driving_expanded_dataset.csv'
    if not os.path.exists(dataset_path):
        dataset_path = 'autonomous_driving_expanded_dataset.csv'
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset not found.")
            return
            
    df = pd.read_csv(dataset_path)
    print(f"  → Loaded {len(df)} rows")
    
    # 2. Feature Engineering
    print("\n2. Performing Feature Engineering...")
    df_feat = perform_feature_engineering(df)
    
    # 3. Perception ML Training
    print("\n3. Training Perception Models...")
    model_nb, model_dt, le_behavior, scaler, model_rf, model_gbr = train_perception_models(df_feat)
    
    # 4. Simulation Execution
    print("\n4. Running Autonomous Simulation Pipeline...")
    
    # Select one representative index for each unique behavior
    demo_indices = []
    unique_behaviors = df_feat['behavior_label'].unique()
    for b in sorted(unique_behaviors):
        # Pick the first occurrence for each behavior
        idx = df_feat[df_feat['behavior_label'] == b].index[0]
        demo_indices.append(idx)
    
    print(f"  → Selected {len(demo_indices)} scenarios representing: {list(sorted(unique_behaviors))}")
    
    for i, idx in enumerate(demo_indices):
        print(f"\n--- Scenario {i+1} (Index: {idx}) ---")
        row = df_feat.iloc[idx]
        
        # A: Perception Inference
        predicted_behavior, risk_score = predict_perception(
            row, model_nb, model_dt, le_behavior, FEATURES,
            scaler=scaler, model_rf=model_rf, model_gbr=model_gbr
        )
        print(f"  > Predicted Behavior: {predicted_behavior} | Risk Score: {risk_score:.4f}")
        
        # B: Knowledge Base Construction (Bayesian + Rules)
        grid, start = create_grid_map(row, risk_score, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT)
        print("  > Grid Map updated via Bayesian and Rule-based models.")
        
        goal = set_goal(row, predicted_behavior, start, grid, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT)
        print(f"  > Target Goal: {goal}")
        
        # C: Path Planning (A* with Fallback)
        path, fsm_cmd, used_fallback = plan_path_with_fallback(grid, start, goal, predicted_behavior)
        
        if fsm_cmd == "brake":
            print("  > [ACTION] Brake - Safe path not found or stop command issued.")
            if path: visualize_scenario(grid, path, start, goal, row, risk_score, idx)
        elif path:
            status = "PLAN (Relaxed)" if used_fallback else "PLAN"
            print(f"  > [SUCCESS] A* Path found ({status})!")
            visualize_scenario(grid, path, start, goal, row, risk_score, idx)
            create_scenario_gif(grid, path, start, goal, row, risk_score, idx)
        else:
            print("  > [FAILED] Pathfinding failed completely. Emergency stop.")

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()

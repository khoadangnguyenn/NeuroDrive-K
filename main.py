import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all necessary modules
from modules.feature_engineering import perform_feature_engineering
from modules.perception_ml import train_perception_models, predict_perception, FEATURES
from modules.bayes import apply_bayesian_update
from modules.rule_based import apply_safety_rules, apply_behavior_rules, set_goal
from modules.knowledge_base import create_grid_map
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
    
    # 2. Feature Engineer
    print("\n[FEATURE ENGINEER] Performing Feature Engineering...")
    df_feat = perform_feature_engineering(df)
    
    # 3. ML Training
    print("\n[ML TRAINING] Training Perception Models (GPU support enabled)...")
    (model_nb, model_dt, le_behavior, scaler, 
     model_rf, model_xgb_cls, model_xgb_reg) = train_perception_models(df_feat)


    
    # 4. Simulation Execution
    print("\n4. Running Autonomous Simulation Pipeline...")
    
    demo_indices = []
    unique_behaviors = df_feat['behavior_label'].unique()
    for b in sorted(unique_behaviors):
        idx = df_feat[df_feat['behavior_label'] == b].index[0]
        demo_indices.append(idx)
    
    print(f"  → Selected {len(demo_indices)} scenarios representing: {list(sorted(unique_behaviors))}")
    
    for i, idx in enumerate(demo_indices):
        print(f"\n--- Scenario {i+1} (Index: {idx}) ---")
        row = df_feat.iloc[idx]
        
        # 4.1 Perception Inference (ML)
        predicted_behavior_ml, risk_score_ml = predict_perception(
            row, model_nb, model_dt, le_behavior, FEATURES,
            scaler=scaler, model_rf=model_rf,
            model_xgb_cls=model_xgb_cls, model_xgb_reg=model_xgb_reg
        )


        print(f"  > [ML] Predicted: {predicted_behavior_ml} | Score: {risk_score_ml:.4f}")

        # 4.2 Bayesian Risk Model
        p_base, p_risk = apply_bayesian_update(
            risk_score_ml, 
            row['weather_condition'], 
            row['road_surface_condition'], 
            row['visibility_range_m']
        )
        print(f"  > [BAYES] Updated Risk: {p_risk:.4f} (Base: {p_base:.4f})")

        # 4.3 Rule-based System
        safety_decision = apply_safety_rules(row)
        if safety_decision:
            final_behavior = safety_decision
            print(f"  > [RULES] Safety override: {final_behavior}")
        else:
            final_behavior = apply_behavior_rules(row)
            print(f"  > [RULES] Tactical decision: {final_behavior}")

        # 4.4 Map Fusion
        grid, start = create_grid_map(row, p_risk, final_behavior, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT)
        
        # Determine the target goal based on rules and road geometry
        goal = set_goal(row, final_behavior, start, grid, GRID_WIDTH, GRID_HEIGHT)
        print(f"  > [MAP FUSION] Cost grid generated. Goal: {goal}")

        # 4.5 Path Planning Algorithm
        path, fsm_cmd, used_fallback = plan_path_with_fallback(grid, start, goal, final_behavior)
        
        if fsm_cmd == "brake":
            print("  > [PATH PLANNING] Action: BRAKE - No safe path.")
        elif path:
            status = "Relaxed" if used_fallback else "Optimal"
            print(f"  > [PATH PLANNING] Success: {status} Path found ({len(path)} nodes).")
        else:
            print("  > [PATH PLANNING] Failed: Emergency Stop.")

        # Visualization & Output
        if path or fsm_cmd == "brake":
            visualize_scenario(grid, path, start, goal, row, p_risk, idx)
            if path:
                create_scenario_gif(grid, path, start, goal, row, p_risk, idx)
        
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()


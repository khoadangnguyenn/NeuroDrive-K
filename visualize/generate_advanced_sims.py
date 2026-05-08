
import os
import sys
import pandas as pd
import joblib

# Fix path
sys.path.append(os.getcwd())

from modules.feature_engineering import perform_feature_engineering
from modules.perception_ml import predict_perception, FEATURES
from modules.knowledge_base import create_grid_map
from modules.rule_based import set_goal
from modules.visualizer import DrivingVisualizer

def main():
    print("=== ADVANCED AUTONOMOUS DRIVING SIMULATION GALLERY ===")
    
    # 1. Load Data
    dataset_path = 'autonomous_driving_expanded_dataset.csv'
    df = pd.read_csv(dataset_path)
    df_feat = perform_feature_engineering(df)
    
    # 2. Load Pre-trained Models
    try:
        model_nb = joblib.load('models/naive_bayes_behavior.pkl')
        model_rf = joblib.load('models/random_forest_behavior.pkl')
        model_dt = joblib.load('models/decision_tree_risk.pkl')
        model_gbr = joblib.load('models/gradient_boosting_risk.pkl')
        le_behavior = joblib.load('models/le_behavior.pkl')
        scaler = joblib.load('models/scaler.pkl')
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run main.py first to train models.")
        return

    # 3. Define Scenarios to Visualize
    # Based on previous find_scenarios.py results:
    scenarios = [
        {"idx": 0, "name": "following_normal", "desc": "Normal Car Following"},
        {"idx": 5, "name": "emergency_stop", "desc": "Obstacle Ahead - Emergency Stop"},
        {"idx": 9, "name": "lane_change", "desc": "Lane Change Maneuver"},
        {"idx": 14, "name": "overtake", "desc": "High Speed Overtake"}
    ]
    
    viz = DrivingVisualizer()
    
    for sc in scenarios:
        idx = sc["idx"]
        name = sc["name"]
        print(f"\nProcessing Scenario: {sc['desc']} (Index {idx})")
        
        row = df_feat.iloc[idx]
        
        # Perception
        predicted_behavior, risk_score = predict_perception(
            row, model_nb, model_dt, le_behavior, FEATURES,
            scaler=scaler, model_rf=model_rf, model_gbr=model_gbr
        )
        
        # Knowledge Base
        grid, start = create_grid_map(row, risk_score)
        
        # Goal Setting
        goal = set_goal(row, predicted_behavior, start, grid)
        
        # Visualization
        output_path = f"results/gallery/{name}.gif"
        viz.create_simulation(row, grid, start, goal, predicted_behavior, risk_score, output_path)

    print("\n" + "="*50)
    print("GALLERY COMPLETE!")
    print("Check 'results/gallery/' for the animations.")
    print("="*50)

if __name__ == "__main__":
    main()

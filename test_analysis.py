import os
import time
import heapq
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.model_selection import train_test_split

# Import project modules
from modules.feature_engineering import perform_feature_engineering
from modules.knowledge_base import create_grid_map
from modules.rule_based import apply_safety_rules, apply_behavior_rules, set_goal
from modules.bayes import apply_bayesian_update
from modules.perception_ml import FEATURES

# =================================================================
# 1. INSTRUMENTED A* ALGORITHM (Standalone)
# =================================================================

def calculate_heuristic(a, b):
    beta = 10 
    return abs(b[1] - a[1]) + beta * abs(b[0] - a[0])

def run_astar_instrumented(grid, start, goal):
    start_time = time.perf_counter()
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    states_explored = 0
    
    directions = [(0, 1), (-1, 1), (1, 1)]
    
    while open_set:
        _, current = heapq.heappop(open_set)
        states_explored += 1
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            end_time = time.perf_counter()
            return path[::-1], states_explored, (end_time - start_time) * 1000 # ms
            
        for dy, dx in directions:
            neighbor = (current[0] + dy, current[1] + dx)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] >= 1e9:
                    continue
                
                move_cost = np.sqrt(dy**2 + dx**2)
                penalty = 5 if dy != 0 else 0
                tentative_g = g_score[current] + move_cost + (grid[neighbor] * 0.01) + penalty
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + calculate_heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    
    end_time = time.perf_counter()
    return None, states_explored, (end_time - start_time) * 1000 # ms

# =================================================================
# 2. ANALYSIS CORE
# =================================================================

def analyze_behavior_prediction(df_feat, models_dir):
    print("\n[6.1.1] Đánh giá mô hình Behavior Prediction (Classification)...")
    
    # Load Necessary Components
    le = joblib.load(os.path.join(models_dir, 'le_behavior.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    
    # Prepare Test Data
    X = df_feat[FEATURES]
    X_scaled = scaler.transform(X)
    y_true = le.transform(df_feat['behavior_label'])
    _, X_test, _, y_test = train_test_split(X_scaled, y_true, test_size=0.2, random_state=42, stratify=y_true)
    
    model_files = {
        'Naive Bayes': 'naive_bayes_behavior.pkl',
        'Random Forest': 'random_forest_behavior.pkl',
        'XGBoost': 'xgb_behavior.pkl'
    }
    
    results = []
    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            model = joblib.load(path)
            y_pred = model.predict(X_test)
            results.append({
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'F1': f1_score(y_test, y_pred, average='macro', zero_division=0)
            })
    
    # Add a Decision Tree Classifier for comparison as requested in example
    from sklearn.tree import DecisionTreeClassifier
    # Split the original data to get a proper training set for this temporary comparison
    X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(
        X_scaled, y_true, test_size=0.2, random_state=42, stratify=y_true
    )
    dt_cls = DecisionTreeClassifier(max_depth=10, random_state=42)
    dt_cls.fit(X_train_temp, y_train_temp)
    y_pred_dt = dt_cls.predict(X_test_temp)
    results.append({
        'Model': 'Decision Tree',
        'Accuracy': accuracy_score(y_test_temp, y_pred_dt),
        'Precision': precision_score(y_test_temp, y_pred_dt, average='macro', zero_division=0),
        'Recall': recall_score(y_test_temp, y_pred_dt, average='macro', zero_division=0),
        'F1': f1_score(y_test_temp, y_pred_dt, average='macro', zero_division=0)
    })
    
    print("\nTable 1.1: Kết quả đánh giá Behavior Prediction")
    print("| Model | Accuracy | Precision | Recall | F1-Score |")
    print("|-------|----------|-----------|--------|----------|")
    for r in sorted(results, key=lambda x: x['F1'], reverse=True):
        print(f"| {r['Model']} | {r['Accuracy']:.1%} | {r['Precision']:.1%} | {r['Recall']:.1%} | {r['F1']:.1%} |")
    
    return results

def analyze_risk_prediction(df_feat, models_dir):
    print("\n[6.1.2] Đánh giá mô hình Risk Prediction (Regression)...")
    
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    X = df_feat[FEATURES]
    X_scaled = scaler.transform(X)
    y_true = df_feat['computed_risk']
    _, X_test, _, y_test = train_test_split(X_scaled, y_true, test_size=0.2, random_state=42)
    
    model_files = {
        'Decision Tree': 'decision_tree_risk.pkl',
        'Gradient Boosting': 'gradient_boosting_risk.pkl',
        'XGBoost': 'xgb_risk.pkl'
    }
    
    results = []
    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            model = joblib.load(path)
            y_pred = model.predict(X_test)
            results.append({
                'Model': name,
                'R2 Score': r2_score(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred)
            })
            
    print("\nTable 1.2: Kết quả đánh giá Risk Prediction")
    print("| Model | R2 Score | MAE | MSE |")
    print("|-------|----------|-----|-----|")
    for r in sorted(results, key=lambda x: x['R2 Score'], reverse=True):
        print(f"| {r['Model']} | {r['R2 Score']:.4f} | {r['MAE']:.4f} | {r['MSE']:.4f} |")
    
    return results

def analyze_astar_performance(df_feat):
    print("\n[6.2] Hiệu suất thuật toán A*...")
    
    scenarios = {
        "Urban Traffic": df_feat[df_feat['traffic_density_veh_per_km'] > 40].head(10),
        "Highway": df_feat[df_feat['speed_limit_kmh'] >= 80].head(10),
        "Foggy Weather": df_feat[df_feat['visibility_range_m'] < 100].head(10)
    }
    
    results = []
    for name, data in scenarios.items():
        if data.empty: continue
        times, states, success_count = [], [], 0
        for _, row in data.iterrows():
            risk_score = row['computed_risk'] if 'computed_risk' in row else 0.5
            grid, start = create_grid_map(row, risk_score, row['behavior_label'], 120, 80)
            goal = set_goal(row, row['behavior_label'], start, grid, 120, 80)
            path, state_count, exec_time = run_astar_instrumented(grid, start, goal)
            if path:
                success_count += 1
                times.append(exec_time)
                states.append(state_count)
        
        results.append({
            'Scenario': name,
            'Time(ms)': np.mean(times) if times else 0,
            'States': int(np.mean(states)) if states else 0,
            'Success Rate': (success_count / len(data)) * 100
        })
        
    print("\nTable 2: Hiệu suất thuật toán A*")
    print("| Scenario | Time(ms) | States | Success Rate |")
    print("|----------|----------|--------|--------------|")
    for r in results:
        print(f"| {r['Scenario']} | {r['Time(ms)']:.1f} | {r['States']} | {r['Success Rate']:.0f}% |")
    
    return results

def analyze_safety_comparison(df_feat):
    print("\n[6.3 & 6.4] Đánh giá an toàn & So sánh kịch bản...")
    
    scenarios = {
        "Normal Weather": df_feat[df_feat['visibility_range_m'] >= 200].head(20),
        "Heavy Traffic": df_feat[df_feat['traffic_density_veh_per_km'] > 50].head(20),
        "Foggy Weather": df_feat[df_feat['visibility_range_m'] < 100].head(20)
    }
    
    results = []
    for name, data in scenarios.items():
        if data.empty: continue
        collision_risks, ade_values = [], []
        for _, row in data.iterrows():
            risk_score = row['computed_risk'] if 'computed_risk' in row else 0.5
            grid, start = create_grid_map(row, risk_score, row['behavior_label'], 120, 80)
            goal = set_goal(row, row['behavior_label'], start, grid, 120, 80)
            path, _, _ = run_astar_instrumented(grid, start, goal)
            
            if path:
                risk_nodes = [grid[node] for node in path if grid[node] > 100]
                collision_risks.append(len(risk_nodes) / len(path))
                lane_center = start[0]
                deviations = [abs(node[0] - lane_center) for node in path]
                ade_values.append(np.mean(deviations) * 0.02)
            else:
                collision_risks.append(1.0)
                ade_values.append(1.5)
                
        avg_ade = np.mean(ade_values)
        stability = "High" if avg_ade < 0.2 else "Medium"
        if avg_ade > 0.4: stability = "Low"

        results.append({
            'Scenario': name,
            'Collision Rate': np.mean(collision_risks),
            'ADE': avg_ade,
            'Stability': stability
        })
        
    print("\nTable 3: So sánh các kịch bản")
    print("| Scenario | Collision Rate | ADE | Stability |")
    print("|----------|----------------|-----|-----------|")
    for r in results:
        print(f"| {r['Scenario']} | {r['Collision Rate']:.1%} | {r['ADE']:.2f} | {r['Stability']} |")

# =================================================================
# MAIN EXECUTION
# =================================================================

def main():
    print("="*60)
    print("  AUTONOMOUS DRIVING - COMPREHENSIVE RESULTS ANALYSIS")
    print("="*60)
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, 'data', 'autonomous_driving_expanded_dataset.csv')
    models_dir = os.path.join(project_root, 'models')
    
    if not os.path.exists(data_path):
        data_path = 'autonomous_driving_expanded_dataset.csv'
    
    df = pd.read_csv(data_path)
    df_feat = perform_feature_engineering(df)
    
    # 6.1 ML Analysis
    analyze_behavior_prediction(df_feat, models_dir)
    analyze_risk_prediction(df_feat, models_dir)
    
    # 6.2 A* Analysis
    analyze_astar_performance(df_feat)
    
    # 6.3 & 6.4 Safety Comparison
    analyze_safety_comparison(df_feat)
    
    print("\n" + "="*60)
    print("  ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()

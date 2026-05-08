from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import os
import pandas as pd
from modules.visualizer import compute_shap_explanation

FEATURES = [
    # Raw features
    'obstacle_distance_m', 'relative_speed_mps', 'num_obstacles', 'lane_offset_m',
    'traffic_density_veh_per_km', 'road_curvature_1pm', 'road_width_m',
    'speed_limit_kmh', 'ego_speed_mps', 'ego_acceleration_mps2', 'steering_angle_deg',
    'yaw_rate_rads', 'throttle_position', 'brake_pressure', 'visibility_range_m',
    # Encoded categoricals
    'weather_encoded', 'road_encoded',
    # Core engineered features
    'closing_rate_mps', 'ttc', 'abs_ttc', 'ego_speed_kmh', 'speed_over_limit_kmh',
    'curvature_abs', 'brake_throttle_diff', 'kinetic_danger', 'safety_ratio', 
    'visibility_factor', 'centrifugal_risk', 'braking_urgency',
    # Interaction features
    'deceleration_needed', 'lane_danger', 
    'weather_visibility', 'congestion_index',
]


def train_perception_models(df):
    """
    Handles ML requirements (Naive Bayes, Decision Tree) and Ensemble optimizations.
    
    Models:
    - Naive Bayes: Behavior prediction (classification)
    - Decision Tree: Risk assessment (regression)
    - RandomForest & GBR: Ensemble improvements for robustness
    
    Optimizations:
    - StandardScaler for feature normalization
    - Class weight balancing for behavior distribution
    - StratifiedKFold cross-validation
    - SHAP Explainability for model transparency
    """
    print("Training Perception Models (Optimized)...")
    print(f"  → Using {len(FEATURES)} features")
    
    X = df[FEATURES].copy()
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES, index=X.index)
    print(f"  → StandardScaler fitted")
    
    # (D) Behavior Prediction (Classification)
    print("\n  --- Behavior Prediction (Classification) ---")
    
    le_behavior = LabelEncoder()
    y_behavior = le_behavior.fit_transform(df['behavior_label'])
    class_names = le_behavior.classes_
    print(f"  → Classes: {list(class_names)}")
    
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X_scaled, y_behavior, test_size=0.2, random_state=42, stratify=y_behavior
    )
    
    print("\n  [1] Naive Bayes Training:")
    param_grid_nb = {'var_smoothing': np.logspace(0, -9, num=20)}
    cv_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_nb = GridSearchCV(
        GaussianNB(), param_grid_nb, cv=cv_strat, 
        scoring='f1_macro', n_jobs=-1
    )
    grid_nb.fit(X_train_cls, y_train_cls)
    model_nb = grid_nb.best_estimator_
    
    y_pred_nb = model_nb.predict(X_test_cls)
    acc_nb = model_nb.score(X_test_cls, y_test_cls)
    print(f"    Accuracy: {acc_nb:.4f}")
    
    # [2] RandomForest Ensemble
    print("  [2] RandomForest Ensemble Training:")
    param_grid_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced']
    }
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42), param_grid_rf, 
        cv=cv_strat, scoring='f1_macro', n_jobs=-1
    )
    grid_rf.fit(X_train_cls, y_train_cls)
    model_rf = grid_rf.best_estimator_
    
    y_pred_rf = model_rf.predict(X_test_cls)
    acc_rf = model_rf.score(X_test_cls, y_test_cls)
    print(f"    Accuracy: {acc_rf:.4f}")
    
    # Choose best behavior model
    if acc_rf >= acc_nb:
        model_behavior = model_rf
        print(f"\n  → Using RandomForest for behavior prediction")
    else:
        model_behavior = model_nb
        print(f"\n  → Using Naive Bayes for behavior prediction")
    
    # (E) Risk Assessment (Regression)
    print("\n  --- Risk Assessment (Regression) ---")
    
    y_risk = df['computed_risk']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_scaled, y_risk, test_size=0.2, random_state=42
    )
    
    # [3] Decision Tree Regressor
    print("\n  [3] Decision Tree Regressor Training:")
    param_grid_dt = {
        'max_depth': [5, 7, 10, 15],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10]
    }
    grid_dt = GridSearchCV(
        DecisionTreeRegressor(random_state=42), param_grid_dt, 
        cv=5, scoring='r2', n_jobs=-1
    )
    grid_dt.fit(X_train_reg, y_train_reg)
    model_dt = grid_dt.best_estimator_
    
    y_pred_dt = model_dt.predict(X_test_reg)
    r2_dt = r2_score(y_test_reg, y_pred_dt)
    print(f"    R² Score: {r2_dt:.4f}")
    
    # [4] Gradient Boosting Regressor
    print("\n  [4] Gradient Boosting Regressor Training:")
    param_grid_gbr = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'min_samples_split': [5, 10]
    }
    grid_gbr = GridSearchCV(
        GradientBoostingRegressor(random_state=42), param_grid_gbr,
        cv=5, scoring='r2', n_jobs=-1
    )
    grid_gbr.fit(X_train_reg, y_train_reg)
    model_gbr = grid_gbr.best_estimator_
    
    y_pred_gbr = model_gbr.predict(X_test_reg)
    r2_gbr = r2_score(y_test_reg, y_pred_gbr)
    print(f"    R² Score: {r2_gbr:.4f}")
    
    # Choose best risk model
    if r2_gbr >= r2_dt:
        model_risk = model_gbr
        print(f"\n  → Using Gradient Boosting for risk assessment")
    else:
        model_risk = model_dt
        print(f"\n  → Using Decision Tree for risk assessment")
    
    compute_shap_explanation(model_behavior, X_test_cls, class_names, FEATURES)
    
    # Persistence
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(model_nb, os.path.join(models_dir, 'naive_bayes_behavior.pkl'))
    joblib.dump(model_rf, os.path.join(models_dir, 'random_forest_behavior.pkl'))
    joblib.dump(model_dt, os.path.join(models_dir, 'decision_tree_risk.pkl'))
    joblib.dump(model_gbr, os.path.join(models_dir, 'gradient_boosting_risk.pkl'))
    joblib.dump(le_behavior, os.path.join(models_dir, 'le_behavior.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    print(f"\n  → Models saved to {models_dir}")
    
    return model_nb, model_dt, le_behavior, scaler, model_rf, model_gbr




def predict_perception(row, model_nb, model_dt, le_behavior, features, scaler=None,
                       model_rf=None, model_gbr=None):
    """
    Predicts behavior and risk score for a given scenario row.
    
    Uses ensemble models if available (RF for behavior, GBR for risk).
    """
    X_single = pd.DataFrame([row[features]], columns=features)
    
    if scaler is not None:
        X_single = pd.DataFrame(scaler.transform(X_single), columns=features)
    
    if model_rf is not None:
        behavior_idx = model_rf.predict(X_single)[0]
    else:
        behavior_idx = model_nb.predict(X_single)[0]
    predicted_behavior = le_behavior.inverse_transform([behavior_idx])[0]
    
    if model_gbr is not None:
        risk_score = model_gbr.predict(X_single)[0]
    else:
        risk_score = model_dt.predict(X_single)[0]
    
    risk_score = np.clip(risk_score, 0, 1)
    
    return predicted_behavior, risk_score

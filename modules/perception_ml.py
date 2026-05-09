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
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
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
    - RandomForest & XGBoost: Ensemble improvements for robustness

    
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
    
    # [2.1] XGBoost Classification
    if XGB_AVAILABLE:
        print("\n  [2.1] XGBoost Classification Training:")
        try:
            tree_method = 'gpu_hist'
            device = 'cuda'
            
            model_xgb_cls = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                tree_method='gpu_hist', # Standard for many Colab environments
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
            model_xgb_cls.fit(X_train_cls, y_train_cls)
            acc_xgb = model_xgb_cls.score(X_test_cls, y_test_cls)
            print(f"    XGBoost Accuracy: {acc_xgb:.4f}")
            if acc_xgb > acc_rf:
                model_behavior = model_xgb_cls
                print("    → XGBoost outperformed RandomForest, using XGBoost.")
        except Exception as e:
            print(f"    → XGBoost training failed: {e}")
            model_xgb_cls = None
    else:
        model_xgb_cls = None

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
    
    # [4] XGBoost Regression
    if XGB_AVAILABLE:
        print("\n  [4] XGBoost Regression Training:")
        try:
            model_xgb_reg = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.1,
                tree_method='gpu_hist',
                random_state=42
            )
            model_xgb_reg.fit(X_train_reg, y_train_reg)
            r2_xgb = r2_score(y_test_reg, model_xgb_reg.predict(X_test_reg))
            print(f"    XGBoost R² Score: {r2_xgb:.4f}")
            if r2_xgb > r2_dt:
                model_risk = model_xgb_reg
                print("    → XGBoost outperformed Decision Tree, using XGBoost.")
            else:
                model_risk = model_dt
        except Exception as e:
            print(f"    → XGBoost training failed (falling back to Decision Tree): {e}")
            model_risk = model_dt
            model_xgb_reg = None
    else:
        model_risk = model_dt
        model_xgb_reg = None
    
    compute_shap_explanation(model_behavior, X_test_cls, class_names, FEATURES)
    
    # Persistence
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, 'models')
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(model_nb, os.path.join(models_dir, 'naive_bayes_behavior.pkl'))
    joblib.dump(model_rf, os.path.join(models_dir, 'random_forest_behavior.pkl'))
    joblib.dump(model_dt, os.path.join(models_dir, 'decision_tree_risk.pkl'))
    if model_xgb_cls: joblib.dump(model_xgb_cls, os.path.join(models_dir, 'xgb_behavior.pkl'))
    if model_xgb_reg: joblib.dump(model_xgb_reg, os.path.join(models_dir, 'xgb_risk.pkl'))
    joblib.dump(le_behavior, os.path.join(models_dir, 'le_behavior.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    print(f"\n  → Models saved to {models_dir}")
    
    return model_nb, model_dt, le_behavior, scaler, model_rf, model_xgb_cls, model_xgb_reg






def predict_perception(row, model_nb, model_dt, le_behavior, features, scaler=None,
                       model_rf=None, model_xgb_cls=None, model_xgb_reg=None):
    """
    Predicts behavior and risk score for a given scenario row.
    
    Uses ensemble models if available (XGB > RF for behavior, XGB > DT for risk).
    """
    X_single = pd.DataFrame([row[features]], columns=features)
    
    if scaler is not None:
        X_single = pd.DataFrame(scaler.transform(X_single), columns=features)
    
    # 1. Behavior Prediction
    if model_xgb_cls is not None:
        behavior_idx = model_xgb_cls.predict(X_single)[0]
    elif model_rf is not None:
        behavior_idx = model_rf.predict(X_single)[0]
    else:
        behavior_idx = model_nb.predict(X_single)[0]
    predicted_behavior = le_behavior.inverse_transform([behavior_idx])[0]
    
    # 2. Risk Prediction
    if model_xgb_reg is not None:
        risk_score = model_xgb_reg.predict(X_single)[0]
    else:
        risk_score = model_dt.predict(X_single)[0]
    
    risk_score = np.clip(risk_score, 0, 1)
    
    return predicted_behavior, risk_score



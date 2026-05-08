import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def perform_feature_engineering(df):
    """
    Transforms raw CSV data into physics-based features representing driving danger.
    
    Processing Steps:
    - Data Cleaning: Handling duplicates, missing values, and clipping extreme bounds.
    - Feature Creation: Engineering physics-informed metrics (TTC, Kinetic Danger, Centrifugal Risk).
    - Interaction Features: Combining metrics like weather-visibility and congestion indices.
    - Computed Risk: Creating a robust ground-truth target based on physics rather than noisy raw labels.
    """
    print("Performing Feature Engineering (Optimized)...")
    df_feat = df.drop_duplicates().copy()

    # 1. Basic Cleaning & Type Conversion
    cat_cols = ['weather_condition', 'road_surface_condition', 'behavior_label']
    num_cols = [
        'obstacle_distance_m','relative_speed_mps','num_obstacles','lane_offset_m',
        'traffic_density_veh_per_km','road_curvature_1pm','road_width_m',
        'speed_limit_kmh','ego_speed_mps','ego_acceleration_mps2','steering_angle_deg',
        'yaw_rate_rads','throttle_position','brake_pressure','visibility_range_m'
    ]
    
    for c in cat_cols:
        if c in df_feat.columns:
            df_feat[c] = df_feat[c].astype(str).str.strip().str.lower()
    
    for c in num_cols:
        if c in df_feat.columns:
            df_feat[c] = pd.to_numeric(df_feat[c], errors='coerce')
            # Fill missing with median
            df_feat[c] = df_feat[c].fillna(df_feat[c].median())

    for c in cat_cols:
        if c in df_feat.columns:
            mode_val = df_feat[c].mode(dropna=True)
            df_feat[c] = df_feat[c].fillna(mode_val.iloc[0] if len(mode_val) else 'unknown')

    # 2. Physical Clipping (Standardization Bounds)
    bounds = {
        'obstacle_distance_m': (0, 300),
        'num_obstacles': (0, 20),
        'traffic_density_veh_per_km': (0, 300),
        'road_width_m': (2.0, 8.0),
        'speed_limit_kmh': (0, 150),
        'ego_speed_mps': (0, 60),
        'throttle_position': (0, 1),
        'brake_pressure': (0, 1),
        'visibility_range_m': (0, 500)
    }
    for c, (lo, hi) in bounds.items():
        if c in df_feat.columns:
            df_feat[c] = df_feat[c].clip(lo, hi)
    
    if 'num_obstacles' in df_feat.columns:
        df_feat['num_obstacles'] = df_feat['num_obstacles'].round().astype(int)

    # 3. Categorical Encoding
    le_weather = LabelEncoder()
    df_feat['weather_encoded'] = le_weather.fit_transform(df_feat['weather_condition'])
    
    le_road = LabelEncoder()
    df_feat['road_encoded'] = le_road.fit_transform(df_feat['road_surface_condition'])
    
    # 4. Feature Engineering
    df_feat['closing_rate_mps'] = np.maximum(df_feat['relative_speed_mps'], 0)
    df_feat['ttc'] = df_feat['obstacle_distance_m'] / (df_feat['closing_rate_mps'] + 1e-3)
    df_feat['ttc'] = df_feat['ttc'].clip(0, 120)
    df_feat['abs_ttc'] = df_feat['ttc']
    
    # Ego speed in km/h and over limit
    df_feat['ego_speed_kmh'] = df_feat['ego_speed_mps'] * 3.6
    df_feat['speed_over_limit_kmh'] = df_feat['ego_speed_kmh'] - df_feat['speed_limit_kmh']
    
    # Absolute road curvature
    df_feat['curvature_abs'] = df_feat['road_curvature_1pm'].abs()

    # Braking interaction
    df_feat['brake_throttle_diff'] = df_feat['brake_pressure'] - df_feat['throttle_position']

    # Kinetic Danger (Energy-based risk)
    df_feat['kinetic_danger'] = df_feat['ego_speed_mps'] * df_feat['traffic_density_veh_per_km']
    
    # Safety Distance Ratio
    df_feat['safety_ratio'] = df_feat['obstacle_distance_m'] / np.maximum(df_feat['ego_speed_mps'], 1.0)
    
    # Visibility Factor (Scaled 0-1)
    df_feat['visibility_factor'] = df_feat['visibility_range_m'] / 500.0
    
    # Centrifugal Risk (Curvature × Velocity²)
    df_feat['centrifugal_risk'] = df_feat['curvature_abs'] * (df_feat['ego_speed_mps']**2)
    
    # Braking Urgency
    delta_v = np.maximum(0, df_feat['ego_speed_mps'] - df_feat['relative_speed_mps'])
    df_feat['braking_urgency'] = (delta_v**2) / (2 * np.maximum(0.1, df_feat['obstacle_distance_m']))
    
    # Interaction: Required Deceleration
    rel_speed_sq = df_feat['relative_speed_mps'] ** 2
    df_feat['deceleration_needed'] = rel_speed_sq / (2 * np.maximum(0.1, df_feat['obstacle_distance_m']))
    
    # Lane Danger (Offset × Speed)
    df_feat['lane_danger'] = np.abs(df_feat['lane_offset_m']) * df_feat['ego_speed_mps']
    
    # Weather-Visibility Interaction
    df_feat['weather_visibility'] = df_feat['weather_encoded'] * (1 - df_feat['visibility_range_m'] / 500.0)
    
    # Congestion Index
    df_feat['congestion_index'] = df_feat['traffic_density_veh_per_km'] * df_feat['num_obstacles']
    
    # 5. Computed Risk (Physics-based target combination)
    risk_proximity = 1.0 - np.clip(df_feat['obstacle_distance_m'] / 100.0, 0, 1)
    risk_speed = np.clip(df_feat['ego_speed_mps'] / 30.0, 0, 1)
    risk_density = np.clip(df_feat['traffic_density_veh_per_km'] / 50.0, 0, 1)
    risk_obstacles = np.clip(df_feat['num_obstacles'] / 14.0, 0, 1)
    risk_braking = np.clip(df_feat['braking_urgency'] / (df_feat['braking_urgency'].quantile(0.99) + 1e-6), 0, 1)
    
    # Weighted combination for high-fidelity risk target
    df_feat['computed_risk'] = (
        0.30 * risk_proximity +
        0.20 * risk_speed +
        0.20 * risk_density +
        0.15 * risk_obstacles +
        0.15 * risk_braking
    )
    
    if 'risk_probability' in df_feat.columns:
        df_feat = df_feat.drop(columns=['risk_probability'])
    
    new_features = [
        'closing_rate_mps', 'ttc', 'abs_ttc', 'ego_speed_kmh', 'speed_over_limit_kmh',
        'curvature_abs', 'brake_throttle_diff', 'kinetic_danger', 'safety_ratio', 
        'visibility_factor', 'centrifugal_risk', 'braking_urgency', 'deceleration_needed',
        'lane_danger', 'weather_visibility', 'congestion_index', 'computed_risk'
    ]
    print(f"  → Created {len(new_features)} engineered features.")

    # Save processed dataset
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    features_dir = os.path.join(project_root, 'features')
    os.makedirs(features_dir, exist_ok=True)
    
    try:
        h5_path = os.path.join(features_dir, 'processed_features.h5')
        df_feat.to_hdf(h5_path, key='df', mode='w')
        print(f"  → Features saved to {h5_path}")
    except Exception:
        csv_path = os.path.join(features_dir, 'processed_features.csv')
        print(f"  → Saving to CSV (HDF5 support not found): {csv_path}")
        df_feat.to_csv(csv_path, index=False)
        
    return df_feat

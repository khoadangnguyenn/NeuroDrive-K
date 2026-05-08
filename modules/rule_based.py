import numpy as np

INF = 1e9

# Safety thresholds
MIN_SAFE_DISTANCE_M = 10.0      # Minimum distance for emergency stop (m)
CRITICAL_TTC_S = 2.5            # Critical Time-to-Collision (s)
MIN_VISIBILITY_M = 30.0         # Minimum safe visibility range (m)

# Behavior thresholds
MAX_DENSITY_OVERTAKE = 20.0     # Max traffic density for overtaking (veh/km)
MAX_CURVATURE_OVERTAKE = 0.02   # Max curvature for safe overtaking (1/m)
MIN_ROAD_WIDTH_OVERTAKE = 3.5   # Min road width required for overtaking (m)
HIGH_RISK_THRESHOLD = 0.7       # High risk threshold to yield

def kmh_to_mps(speed_kmh: float) -> float:
    return speed_kmh / 3.6

def obstacle_penalty_profile(label: str):
    """
    Returns penalty profile multipliers based on intended behavior.
    """
    profiles = {
        "follow": {"near_mult": 1.35, "sigma_mult": 0.95, "front_band_mult": 1.25},
        "lane_change": {"near_mult": 1.10, "sigma_mult": 1.05, "front_band_mult": 0.95},
        "overtake": {"near_mult": 0.95, "sigma_mult": 1.15, "front_band_mult": 0.85},
        "yield": {"near_mult": 1.45, "sigma_mult": 0.90, "front_band_mult": 1.40},
        "stop": {"near_mult": 1.60, "sigma_mult": 0.85, "front_band_mult": 1.60},
    }
    return profiles.get(label, {"near_mult": 1.2, "sigma_mult": 1.0, "front_band_mult": 1.0})

def behavior_policy(label: str, start_y: int, grid_w: int):
    """
    Defines cost weights and target offsets based on behavior label.
    """
    if label == "stop":
        return {"goal_x_offset": 0, "w_dist": 0.05, "w_risk": 3.2, "w_lane": 2.6}
    if label == "follow":
        return {"goal_x_offset": -14, "w_dist": 1.0, "w_risk": 2.1, "w_lane": 2.0}
    if label == "yield":
        return {"goal_x_offset": -18, "w_dist": 0.7, "w_risk": 2.8, "w_lane": 2.6}
    if label == "lane_change":
        return {"goal_x_offset": -14, "w_dist": 1.0, "w_risk": 1.9, "w_lane": 0.9}
    if label == "overtake":
        return {"goal_x_offset": -10, "w_dist": 1.3, "w_risk": 1.3, "w_lane": 0.6}
    return {"goal_x_offset": -10, "w_dist": 1.0, "w_risk": 1.8, "w_lane": 1.0}

def apply_safety_rules(row: dict) -> str:
    """
    Applies hard safety constraints.
    """
    obs_distance = float(row['obstacle_distance_m'])
    rel_speed = float(row['relative_speed_mps'])
    ego_speed = float(row['ego_speed_mps'])
    speed_limit_mps = kmh_to_mps(float(row['speed_limit_kmh']))
    
    weather = row.get('weather_condition', 'clear')
    surface = row.get('road_surface_condition', 'dry')
    visibility = float(row['visibility_range_m'])

    # 1. Emergency Collision Avoidance
    if obs_distance < MIN_SAFE_DISTANCE_M:
        return 'stop'
    
    if rel_speed > 0:
        ttc = obs_distance / rel_speed
        if ttc < CRITICAL_TTC_S:
            return 'stop'

    # 2. Visibility Rules
    if visibility < MIN_VISIBILITY_M:
        return 'stop'

    # 3. Surface & Speed Limit Adjustment
    safe_speed_limit = speed_limit_mps
    if surface == 'icy':
        safe_speed_limit *= 0.5
    elif surface == 'wet' or weather == 'rain':
        safe_speed_limit *= 0.8
    
    if ego_speed > safe_speed_limit:
        return 'yield'

    return None

def apply_behavior_rules(row: dict) -> str:
    """
    Determines tactical behavior decisions.
    """
    obs_distance = float(row['obstacle_distance_m'])
    rel_speed = float(row['relative_speed_mps'])
    traffic_density = float(row.get('traffic_density_veh_per_km', 0.0))
    curvature = abs(float(row.get('road_curvature_1pm', 0.0)))
    road_width = float(row.get('road_width_m', 3.0))
    num_obstacles = int(row.get('num_obstacles', 0))
    risk_prob = float(row.get('risk_probability', 0.0))
    
    if risk_prob > HIGH_RISK_THRESHOLD or num_obstacles >= 5:
        return 'yield'

    if obs_distance < 80.0 and rel_speed > 0:
        is_straight = curvature < MAX_CURVATURE_OVERTAKE
        is_wide_enough = road_width >= MIN_ROAD_WIDTH_OVERTAKE
        is_traffic_light = traffic_density < MAX_DENSITY_OVERTAKE
        
        if is_straight and is_wide_enough and is_traffic_light:
            return 'overtake'
        
        can_lane_change = traffic_density < (MAX_DENSITY_OVERTAKE * 1.5) and road_width >= 3.0
        if can_lane_change:
            return 'lane_change'
        
        return 'follow'
        
    return 'follow'

def check_rule_violation(row) -> bool:
    """
    Checks for safety rule violations.
    """
    emergency = apply_safety_rules(row)
    if emergency == 'stop':
        return True
        
    speed_kmh = float(row["ego_speed_mps"]) * 3.6
    basic_violation = bool(
        (speed_kmh > float(row["speed_limit_kmh"]) + 5)
        or (float(row["obstacle_distance_m"]) < 8 and float(row["relative_speed_mps"]) > 2)
        or (abs(float(row["lane_offset_m"])) > float(row["road_width_m"]) * 0.45)
    )
    return basic_violation

def set_goal(row, label, start, grid, grid_width, grid_height):
    """
    Sets target goal coordinates based on behavior and road geometry.
    Used by the main pipeline.
    """
    goal_x = grid_width - 10
    
    curvature = float(row["road_curvature_1pm"])
    road_width_m = float(row["road_width_m"])
    
    start_y, start_x = start
    
    # Estimate centerline position at goal_x
    x_rel = max(goal_x - start_x, 0)
    centerline_goal_y = int(np.clip(start_y + np.clip(curvature * (x_rel**2) * 0.6, -14, 14), 0, grid_height - 1))
    
    lane_shift = max(1, int(round(road_width_m * 0.5)))
    
    if label == "stop":
        return start
    if label == "yield":
        return (centerline_goal_y, goal_x)
    if label == "lane_change":
        return (int(np.clip(centerline_goal_y - lane_shift, 0, grid_height - 1)), goal_x)
    if label == "overtake":
        return (int(np.clip(centerline_goal_y - (lane_shift + 1), 0, grid_height - 1)), goal_x)
        
    return (centerline_goal_y, goal_x)

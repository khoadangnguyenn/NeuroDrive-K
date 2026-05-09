import numpy as np
from .bayes import apply_bayesian_update
from .rule_based import obstacle_penalty_profile, behavior_policy, check_rule_violation, INF

def create_grid_map(row, p_risk, behavior_label, grid_width=120, grid_height=80):
    """
    Builds a Grid Cost Map based on:
    - Road Curvature and Width
    - Obstacle positions and Bayesian dynamic risk (passed via p_risk)
    - Safety Rules (Hard constraints)
    """
    grid_h, grid_w = grid_height, grid_width
    
    # 1. Bayesian Risk Update (NOW PASSED AS ARGUMENT)
    # p_base, p_risk = apply_bayesian_update(...) 
    
    # 2. Determine Starting Position
    base_start_x = 10
    lane_offset_m = float(row["lane_offset_m"])
    curvature = float(row["road_curvature_1pm"])
    road_width_m = float(row["road_width_m"])
    label = str(behavior_label)
    
    start_y = int(np.clip(grid_h // 2 + lane_offset_m * 3.2, 8, grid_h - 9))
    start = (start_y, base_start_x)
    
    # 3. Determine Goal Position based on behavior
    pol = behavior_policy(label, start_y, grid_w)
    goal_x = grid_w + pol["goal_x_offset"]
    if goal_x >= grid_w: goal_x = grid_w - 1
    
    x_rel = max(goal_x - base_start_x, 0)
    # Centerline bends according to curvature
    centerline_goal_y = int(np.clip(start_y + np.clip(curvature * (x_rel**2) * 0.6, -14, 14), 8, grid_h - 9))
    
    lane_shift = max(4, min(10, int(round(road_width_m * 1.5))))
    
    if label == "stop":
        goal = start
    elif label == "lane_change":
        goal = (int(np.clip(centerline_goal_y - lane_shift, 8, grid_h - 9)), goal_x)
    elif label == "overtake":
        goal = (int(np.clip(centerline_goal_y - (lane_shift + 1), 8, grid_h - 9)), goal_x)
    else:
        goal = (centerline_goal_y, goal_x)
        
    # 4. Construct Cost Matrix
    Y, X = np.indices((grid_h, grid_w))
    x_axis = np.arange(grid_w)
    x_rel_1d = np.clip(x_axis - start[1], a_min=0, a_max=None)
    
    # Curvature centerline for the whole grid
    centerline_1d = start[0] + np.clip(curvature * (x_rel_1d**2) * 0.6, -14, 14)
    centerline_2d = centerline_1d[None, :]
    off_center = np.abs(Y - centerline_2d)
    
    cells_per_meter_y = 3.2
    half_width_cells = max(3.0, road_width_m * 0.5 * cells_per_meter_y)
    
    # Rule costs (Road boundary & speed)
    rule_cost = np.zeros((grid_h, grid_w), dtype=float)
    excess = np.maximum(0.0, off_center - half_width_cells)
    rule_cost += (excess**2) * 18.0
    rule_cost[off_center > (half_width_cells + 8.0)] = INF
    rule_cost[:6, :] = INF
    rule_cost[74:, :] = INF
    
    # Distance and lane preference costs
    distance_cost = np.sqrt((Y - goal[0]) ** 2 + (X - goal[1]) ** 2)
    lane_pref_cost = np.abs(Y - goal[0])
    
    # 5. Obstacle Risk modeling
    oy = int(np.clip(start[0] + lane_offset_m * 8, 8, grid_h - 8))
    ox = int(np.clip(start[1] + float(row["obstacle_distance_m"]) * 1.2, 0, grid_w - 1))
    
    density_norm = np.clip(float(row["traffic_density_veh_per_km"]) / 50.0, 0, 1)
    closing_norm = np.clip(max(float(row["relative_speed_mps"]), 0.0) / 15.0, 0, 1)
    obstacles_norm = np.clip(float(row["num_obstacles"]) / 14.0, 0, 1)
    
    amp = p_risk * (170 + 120 * density_norm + 80 * closing_norm)
    profile = obstacle_penalty_profile(label)
    sigma = (6.0 + 6.0 * (1 - obstacles_norm) + 2.0 * (1 - density_norm)) * profile["sigma_mult"]
    
    risk_cost = amp * np.exp(-((Y - oy) ** 2 + (X - ox) ** 2) / (2 * sigma**2))
    
    # Near-field penalty profile
    near_radius = 5
    near_mask = ((Y - oy) ** 2 + (X - ox) ** 2) <= (near_radius**2)
    risk_cost[near_mask] *= profile["near_mult"]
    
    # Front band penalty (area ahead of obstacle)
    fx0, fx1 = ox + 1, min(grid_w, ox + 12)
    fy0, fy1 = max(0, oy - 4), min(grid_h, oy + 5)
    if fx0 < fx1:
        risk_cost[fy0:fy1, fx0:fx1] *= profile["front_band_mult"]
    
    # 5.1 Obstacle Spread Risk (High density spread)
    if obstacles_norm > 0.45:
        spread = int(6 + 12 * obstacles_norm)
        for dy in (-spread, spread):
            oy2 = int(np.clip(oy + dy, 0, grid_h - 1))
            risk_cost += 0.35 * amp * np.exp(-((Y - oy2) ** 2 + (X - ox) ** 2) / (2 * (sigma * 1.2) ** 2))
    
    # 5.2 Speeding Penalty
    speed_kmh = float(row["ego_speed_mps"]) * 3.6
    over_limit = max(0.0, speed_kmh - float(row["speed_limit_kmh"]))
    if over_limit > 0:
        # Penalty area ahead of car if speeding
        f0, f1 = start[1] + 6, min(grid_w, start[1] + 28)
        y0 = max(0, int(centerline_1d.mean() - half_width_cells))
        y1 = min(grid_h, int(centerline_1d.mean() + half_width_cells) + 1)
        rule_cost[y0:y1, f0:f1] += 350 + 45 * over_limit
    
    # 5.3 Stop/Yield Buffer (Forbidden forward zone)
    if label in ["stop", "yield"]:
        rule_cost[max(0, start[0] - 6) : min(grid_h, start[0] + 7), start[1] + 2 : min(grid_w, start[1] + 20)] += 1300.0
        
    # 6. Final Cost Synthesis
    w_risk_eff = pol["w_risk"] * (1 + 0.8 * density_norm)
    total_cost = pol["w_dist"] * distance_cost + w_risk_eff * risk_cost + pol["w_lane"] * lane_pref_cost + rule_cost
    
    # Rule violation check to block the path
    if check_rule_violation(row):
        total_cost[max(0, oy - 2) : min(grid_h, oy + 3), max(0, ox - 5) : min(grid_w, ox + 6)] = INF
    
    return total_cost, start


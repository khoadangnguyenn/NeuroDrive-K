import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import heapq

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.feature_engineering import perform_feature_engineering
from modules.perception_ml import train_perception_models, predict_perception
from modules.knowledge_base import create_grid_map
from modules.rule_based import set_goal

# Constants
GRID_WIDTH = 100
GRID_HEIGHT = 3

def astar_with_frames(grid, start, goal):
    """
    Modified A* that yields exploration states for animation.
    """
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    explored = []
    
    directions = [(0, 1), (-1, 1), (1, 1)]
    
    while open_set:
        _, current = heapq.heappop(open_set)
        explored.append(current)
        
        # Yield state for animation (limiting frequency)
        if len(explored) % 5 == 0:
            yield {"type": "search", "explored": list(explored), "current": current, "path": None}
            
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            final_path = path[::-1]
            yield {"type": "path_found", "explored": list(explored), "current": goal, "path": final_path}
            return
            
        for dy, dx in directions:
            neighbor = (current[0] + dy, current[1] + dx)
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] >= 1000: continue
                move_cost = np.sqrt(dy**2 + dx**2)
                penalty = 5 if dy != 0 else 0
                tentative_g = g_score[current] + move_cost + grid[neighbor] + penalty
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + (abs(goal[1] - neighbor[1]) + 10 * abs(goal[0] - neighbor[0]))
                    heapq.heappush(open_set, (f_score, neighbor))

def generate_gif():
    print("Initializing GIF Generation...")
    df = pd.read_csv('autonomous_driving_expanded_dataset.csv')
    df_feat = perform_feature_engineering(df)
    
    # Train models once
    model_nb, model_dt, le_behavior = train_perception_models(df_feat)
    features = [
        'obstacle_distance_m', 'relative_speed_mps', 'num_obstacles', 'lane_offset_m',
        'traffic_density_veh_per_km', 'road_curvature_1pm', 'road_width_m',
        'speed_limit_kmh', 'ego_speed_mps', 'ego_acceleration_mps2', 'steering_angle_deg',
        'yaw_rate_rads', 'throttle_position', 'brake_pressure', 'visibility_range_m',
        'weather_encoded', 'road_encoded', 'ttc', 'kinetic_danger', 
        'safety_ratio', 'visibility_factor', 'centrifugal_risk', 'braking_urgency'
    ]
    
    # Pick Scenario 5
    idx = 5
    row = df_feat.iloc[idx]
    predicted_behavior, risk_score = predict_perception(row, model_nb, model_dt, le_behavior, features)
    grid, start = create_grid_map(row, risk_score, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT)
    goal = set_goal(row, predicted_behavior, start, grid, grid_width=GRID_WIDTH, grid_height=GRID_HEIGHT)
    
    # Collect Frames
    frames_data = []
    
    # Stage 1: Cost Map visualization (just a few frames)
    for _ in range(5):
        frames_data.append({"type": "init", "grid": grid, "start": start, "goal": goal, "row": row})
        
    # Stage 2: A* Search
    search_gen = astar_with_frames(grid, start, goal)
    final_path = None
    for frame in search_gen:
        frame.update({"grid": grid, "start": start, "goal": goal, "row": row})
        frames_data.append(frame)
        if frame["type"] == "path_found":
            final_path = frame["path"]
            
    # Stage 3: Driving along path
    if final_path:
        for i in range(len(final_path)):
            frames_data.append({
                "type": "drive", 
                "grid": grid, 
                "start": start, 
                "goal": goal, 
                "row": row, 
                "current_pos": final_path[i],
                "path": final_path
            })
            
    # Setup Plot
    fig, ax = plt.subplots(figsize=(15, 6), facecolor='#0f172a')
    plt.tight_layout()
    
    def update(i):
        ax.clear()
        data = frames_data[i]
        grid = data["grid"]
        row = data["row"]
        
        # Style
        ax.set_facecolor('#0f172a')
        ax.imshow(grid, cmap='hot', origin='upper', alpha=0.7, extent=[0, GRID_WIDTH, 2.5, -0.5])
        
        # Grid boundaries
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Lane 0', 'Lane 1', 'Lane 2'], color='white')
        ax.tick_params(colors='white')
        
        # Obstacle
        obs_x = int(np.clip(row['obstacle_distance_m'], 0, GRID_WIDTH - 1))
        ax.scatter(obs_x, start[0], color='#ff4757', s=300, marker='X', edgecolors='white', label='Obstacle')
        
        # Goal
        ax.scatter(data["goal"][1], data["goal"][0], color='#eccc68', s=400, marker='*', edgecolors='white', label='Goal')
        
        if data["type"] == "init":
            ax.scatter(data["start"][1], data["start"][0], color='#2ed573', s=300, marker='>', edgecolors='white', label='Ego (Initial)')
            ax.set_title("1. Environment Perception & Risk Analysis", color='white', fontsize=16, pad=20)
            
        elif data["type"] == "search":
            # Show explored nodes
            ex_y, ex_x = zip(*data["explored"])
            ax.scatter(ex_x, ex_y, color='#70a1ff', s=10, alpha=0.3)
            ax.scatter(data["current"][1], data["current"][0], color='#1e90ff', s=50, marker='o')
            ax.scatter(data["start"][1], data["start"][0], color='#2ed573', s=300, marker='>', edgecolors='white')
            ax.set_title(f"2. A* Pathfinding: Exploring Safe Nodes... ({len(data['explored'])} steps)", color='white', fontsize=16, pad=20)
            
        elif data["type"] == "path_found" or data["type"] == "drive":
            path = data["path"]
            py, px = zip(*path)
            ax.plot(px, py, color='#1e90ff', linewidth=3, alpha=0.8, linestyle='--')
            
            if data["type"] == "drive":
                pos = data["current_pos"]
                ax.scatter(pos[1], pos[0], color='#2ed573', s=400, marker='>', edgecolors='white', label='Ego Moving')
                ax.set_title(f"3. Executing Safe Trajectory | Risk Score: {risk_score:.4f}", color='white', fontsize=16, pad=20)
            else:
                ax.scatter(data["start"][1], data["start"][0], color='#2ed573', s=300, marker='>', edgecolors='white')
                ax.set_title("Path Found! Planning Trajectory...", color='white', fontsize=16, pad=20)

        ax.legend(loc='upper left', facecolor='#1e293b', edgecolor='white', labelcolor='white')
        ax.set_xlim(-5, GRID_WIDTH + 5)
        ax.set_ylim(2.8, -0.8)
        
    print(f"Creating animation with {len(frames_data)} frames...")
    ani = FuncAnimation(fig, update, frames=len(frames_data), interval=100)
    
    os.makedirs('results', exist_ok=True)
    writer = PillowWriter(fps=10)
    ani.save('results/simulation.gif', writer=writer)
    plt.close()
    print("Animation saved to results/simulation.gif")

if __name__ == "__main__":
    generate_gif()

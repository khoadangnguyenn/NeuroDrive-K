
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import heapq
import pandas as pd
import io
from PIL import Image

class DrivingVisualizer:
    def __init__(self, grid_width=100, grid_height=3):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.fps = 15
        
    def _astar_with_frames(self, grid, start, goal):
        rows, cols = grid.shape
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        explored = []
        
        # Directions: Forward, Forward-Left, Forward-Right
        directions = [(0, 1), (-1, 1), (1, 1)]
        
        while open_set:
            _, current = heapq.heappop(open_set)
            explored.append(current)
            
            # Yield every 10 steps to speed up animation
            if len(explored) % 10 == 0:
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
                    # Add penalty for lane changing to encourage staying in lane
                    lane_penalty = 5.0 if dy != 0 else 0.0
                    
                    tentative_g = g_score[current] + move_cost + grid[neighbor] + lane_penalty
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        # Heuristic: distance to goal with lane bias
                        h = (abs(goal[1] - neighbor[1]) + 15 * abs(goal[0] - neighbor[0]))
                        f_score = tentative_g + h
                        heapq.heappush(open_set, (f_score, neighbor))

    def create_simulation(self, row, grid, start, goal, predicted_behavior, risk_score, output_path):
        print(f"Creating Simulation for {predicted_behavior}...")
        
        frames_data = []
        
        # Stage 1: Static Environment (Pause)
        for _ in range(10):
            frames_data.append({"type": "init", "grid": grid, "start": start, "goal": goal, "row": row})
            
        # Stage 2: A* Search
        search_gen = self._astar_with_frames(grid, start, goal)
        final_path = None
        for frame in search_gen:
            frame.update({"grid": grid, "start": start, "goal": goal, "row": row})
            frames_data.append(frame)
            if frame["type"] == "path_found":
                final_path = frame["path"]
                # Pause at path found
                for _ in range(5): frames_data.append(frame)
                
        # Stage 3: Smooth Driving
        if final_path:
            # Interpolate for smoother motion
            smooth_path = []
            for i in range(len(final_path)-1):
                p1 = final_path[i]
                p2 = final_path[i+1]
                steps = 3
                for s in range(steps):
                    interp = s / steps
                    smooth_path.append((p1[0] + (p2[0]-p1[0])*interp, p1[1] + (p2[1]-p1[1])*interp))
            smooth_path.append(final_path[-1])
            
            for pos in smooth_path:
                frames_data.append({
                    "type": "drive", 
                    "grid": grid, 
                    "start": start, 
                    "goal": goal, 
                    "row": row, 
                    "current_pos": pos,
                    "path": final_path
                })
                
        # Final Pause
        for _ in range(15):
            frames_data.append(frames_data[-1])

        # Plotting
        fig, ax = plt.subplots(figsize=(16, 6), facecolor='#0f172a')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.1)
        
        # Define Colors
        C_BG = '#0f172a'
        C_LANE = '#334155'
        C_EGO = '#22c55e'
        C_OBS = '#ef4444'
        C_GOAL = '#eab308'
        C_PATH = '#3b82f6'
        C_SEARCH = '#64748b'
        
        def update(i):
            ax.clear()
            data = frames_data[i]
            grid = data["grid"]
            row = data["row"]
            
            ax.set_facecolor(C_BG)
            
            # Draw Lanes
            for y in range(self.grid_height + 1):
                ax.axhline(y - 0.5, color=C_LANE, linestyle='-', linewidth=2, alpha=0.5)
            
            # Show Cost Map (Hot Map)
            im = ax.imshow(grid, cmap='magma', origin='upper', alpha=0.4, 
                           extent=[-0.5, self.grid_width-0.5, self.grid_height-0.5, -0.5],
                           aspect='auto', interpolation='gaussian')
            
            # Obstacle (Glow effect)
            obs_x = row['obstacle_distance_m']
            ax.scatter(obs_x, start[0], color=C_OBS, s=400, marker='X', edgecolors='white', 
                       linewidth=2, zorder=10, label='Hazard / Obstacle')
            
            # Goal
            ax.scatter(goal[1], goal[0], color=C_GOAL, s=500, marker='*', edgecolors='white', 
                       linewidth=2, zorder=10, label='Target Destination')
            
            # Dynamic Content
            if data["type"] == "init":
                ax.scatter(start[1], start[0], color=C_EGO, s=400, marker='>', edgecolors='white', label='Ego Vehicle')
                title = f"Phase 1: Perception & Risk Analysis\nPredicted Behavior: {predicted_behavior.upper()} | Risk Score: {risk_score:.4f}"
                
            elif data["type"] == "search":
                ex_y, ex_x = zip(*data["explored"])
                ax.scatter(ex_x, ex_y, color=C_SEARCH, s=15, alpha=0.3, zorder=1)
                ax.scatter(data["current"][1], data["current"][0], color=C_PATH, s=80, marker='o', zorder=2)
                ax.scatter(start[1], start[0], color=C_EGO, s=400, marker='>', edgecolors='white', zorder=10)
                title = f"Phase 2: A* Trajectory Optimization\nSearching safe nodes... ({len(data['explored'])} states)"
                
            elif data["type"] in ["path_found", "drive"]:
                path = data["path"]
                py, px = zip(*path)
                ax.plot(px, py, color=C_PATH, linewidth=4, alpha=0.8, linestyle='-', zorder=5, label='Optimized Path')
                
                if data["type"] == "drive":
                    pos = data["current_pos"]
                    ax.scatter(pos[1], pos[0], color=C_EGO, s=500, marker='>', edgecolors='white', zorder=10, label='Autonomous Execution')
                    title = f"Phase 3: Autonomous Trajectory Execution\nState: {predicted_behavior.title()} | Risk Control Active"
                else:
                    ax.scatter(start[1], start[0], color=C_EGO, s=400, marker='>', edgecolors='white', zorder=10)
                    title = "Path Found! Safety Protocol Verified."

            # Styling
            ax.set_title(title, color='white', fontsize=18, fontweight='bold', pad=30)
            ax.set_xlim(-2, self.grid_width + 2)
            ax.set_ylim(self.grid_height - 0.2, -0.8)
            ax.set_yticks(range(self.grid_height))
            ax.set_yticklabels([f"Lane {i}" for i in range(self.grid_height)], color='white', fontsize=12)
            ax.tick_params(axis='x', colors='white', labelsize=10)
            ax.set_xlabel("Distance (m)", color='white', fontsize=12)
            
            # Legend
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1.15), ncol=4, 
                      facecolor='#1e293b', edgecolor='#334155', labelcolor='white', fontsize=10)

        print(f"Rendering {len(frames_data)} frames...")
        ani = FuncAnimation(fig, update, frames=len(frames_data), interval=1000/self.fps)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        writer = PillowWriter(fps=self.fps)
        ani.save(output_path, writer=writer)
        plt.close()
        print(f"Animation saved to {output_path}")


def visualize_scenario(grid, path, start, goal, row, risk_score, idx, grid_width=120, grid_height=80):
    """
    Visualizes the Grid Map and Planned Path for a specific scenario.
    """
    plt.figure(figsize=(15, 5))
    plt.imshow(grid, cmap='YlOrRd', origin='upper', alpha=0.6)
    
    if path:
        py, px = zip(*path)
        plt.plot(px, py, color='cyan', linewidth=4, label='Safe Path', marker='o', markersize=3, markeredgecolor='blue')
    
    plt.scatter(start[1], start[0], color='lime', s=250, label='Ego Vehicle', marker='>', edgecolors='black', zorder=5)
    plt.scatter(goal[1], goal[0], color='gold', s=300, label='Goal', marker='*', edgecolors='black', zorder=5)
    
    obs_x = int(np.clip(start[1] + float(row["obstacle_distance_m"]) * 1.2, 0, grid_width - 1))
    obs_y = int(np.clip(start[0] + float(row["lane_offset_m"]) * 8, 8, grid_height - 8))
    plt.scatter(obs_x, obs_y, color='red', s=200, label='Obstacle', marker='X', edgecolors='black', zorder=5)
    
    behavior = row.get('behavior_label', 'Unknown')
    plt.title(f"AI Pipeline - Scenario {idx} ({behavior.upper()})\nRisk Score: {risk_score:.4f}")
    plt.xlabel("Distance")
    plt.ylabel("Lane")
    plt.legend(loc='upper right')
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plots_dir = os.path.join(project_root, 'visualize', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    save_path = os.path.join(plots_dir, f'scenario_{idx}.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  > [SAVED] Visualization at {save_path}")


def create_scenario_gif(grid, path, start, goal, row, risk_score, idx, duration=100):
    """
    Generates a GIF animation simulating the vehicle moving along the planned path.
    """
    if not path:
        return
        
    frames = []
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gifs_dir = os.path.join(project_root, 'visualize', 'gifs')
    os.makedirs(gifs_dir, exist_ok=True)
    
    step_size = 1
    if len(path) > 50:
        step_size = 2
        
    print(f"  > Generating GIF frames for Scenario {idx}...", end="", flush=True)
    
    for i in range(0, len(path), step_size):
        curr_pos = path[i]
        
        plt.figure(figsize=(12, 5))
        plt.imshow(grid, cmap='YlOrRd', origin='upper', alpha=0.6)
        
        py, px = zip(*path)
        plt.plot(px, py, color='cyan', linewidth=2, alpha=0.3, ls='--')
        
        plt.scatter(curr_pos[1], curr_pos[0], color='lime', s=200, label='Ego Vehicle', marker='>', edgecolors='black', zorder=5)
        plt.scatter(goal[1], goal[0], color='gold', s=250, label='Goal', marker='*', edgecolors='black', zorder=5)
        
        obs_x = int(np.clip(start[1] + float(row["obstacle_distance_m"]) * 1.2, 0, grid.shape[1] - 1))
        obs_y = int(np.clip(start[0] + float(row["lane_offset_m"]) * 8, 8, grid.shape[0] - 8))
        plt.scatter(obs_x, obs_y, color='red', s=150, label='Obstacle', marker='X', edgecolors='black', zorder=5)
        
        plt.title(f"Animation - Scenario {idx} | Step {i+1}/{len(path)}")
        plt.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close()
        print(".", end="", flush=True)
        
    if frames:
        gif_path = os.path.join(gifs_dir, f'scenario_{idx}.gif')
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f" [DONE]\n  > [SAVED] Animation at {gif_path}")


def compute_shap_explanation(model, X_test, class_names, features):
    """
    Computes SHAP values to explain model predictions.
    """
    try:
        import shap
        print("\n  --- SHAP Explainability ---")
        
        X_shap = X_test.iloc[:100] if len(X_test) > 100 else X_test
        
        if hasattr(model, 'estimators_'):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_shap.iloc[:50])
        
        shap_values = explainer.shap_values(X_shap)
        
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        viz_dir = os.path.join(project_root, 'visualize')
        os.makedirs(viz_dir, exist_ok=True)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        if isinstance(shap_values, list):
            shap_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            mean_shap = np.mean(shap_abs, axis=0)
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            mean_shap = np.mean(np.abs(shap_values), axis=(0, 2))
        else:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        sorted_idx = np.argsort(mean_shap)[::-1][:15]
        
        plt.barh(range(len(sorted_idx)), mean_shap[sorted_idx][::-1])
        plt.yticks(range(len(sorted_idx)), [features[i] for i in sorted_idx][::-1])
        plt.xlabel('Mean |SHAP value|')
        plt.title('SHAP Feature Importance (Top 15)')
        
        plt.tight_layout()
        save_path = os.path.join(viz_dir, 'shap_importance.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  → SHAP plot saved to {save_path}")
        
    except ImportError:
        print("\n  → SHAP library not found. Skipping analysis.")
    except Exception as e:
        print(f"\n  → SHAP computation failed: {e}")


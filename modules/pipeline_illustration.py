
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

class PipelineIllustrator:
    def __init__(self, output_dir="visualize/illustrations"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Premium Color Palette
        self.colors = {
            'bg': '#0f172a',
            'card': '#1e293b',
            'text': '#f8fafc',
            'accent1': '#38bdf8', # Sky Blue
            'accent2': '#818cf8', # Indigo
            'accent3': '#fbbf24', # Amber
            'danger': '#ef4444',  # Red
            'success': '#22c55e', # Green
            'neutral': '#94a3b8'  # Slate
        }
        plt.rcParams['text.color'] = self.colors['text']
        plt.rcParams['axes.labelcolor'] = self.colors['text']
        plt.rcParams['xtick.color'] = self.colors['text']
        plt.rcParams['ytick.color'] = self.colors['text']

    def plot_pipeline_flow(self):
        """Generates a high-quality flowchart of the Autonomous Driving Pipeline."""
        print("Generating Pipeline Flow Diagram...")
        fig, ax = plt.subplots(figsize=(14, 8), facecolor=self.colors['bg'])
        ax.set_facecolor(self.colors['bg'])
        
        # Define nodes
        nodes = [
            ("Data Source", (1, 7), self.colors['neutral']),
            ("Feature Engineering", (4, 7), self.colors['accent1']),
            ("ML Perception\n(XGBoost/RF)", (7, 7), self.colors['accent1']),
            ("Bayesian Risk\nModel", (10, 7), self.colors['accent2']),
            ("Rule-Based\nConstraints", (10, 4), self.colors['accent2']),
            ("Cost Map Fusion", (7, 4), self.colors['accent3']),
            ("A* Path Planning", (4, 4), self.colors['accent3']),
            ("Visualization &\nControl", (1, 4), self.colors['success'])
        ]
        
        # Draw boxes
        box_width = 2.2
        box_height = 1.2
        
        for name, pos, color in nodes:
            rect = patches.FancyBboxPatch(
                (pos[0] - box_width/2, pos[1] - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.1",
                linewidth=2, edgecolor=color, facecolor=self.colors['card'],
                zorder=2
            )
            ax.add_patch(rect)
            ax.text(pos[0], pos[1], name, color='white', weight='bold',
                    ha='center', va='center', fontsize=11, zorder=3)

        # Draw arrows
        arrow_style = "simple,head_length=10,head_width=10,tail_width=2"
        connections = [
            (nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3]),
            (nodes[3], nodes[4]), (nodes[4], nodes[5]), (nodes[5], nodes[6]),
            (nodes[6], nodes[7])
        ]
        
        for (n1_name, n1_pos, n1_c), (n2_name, n2_pos, n2_c) in connections:
            ax.annotate("", xy=n2_pos, xytext=n1_pos,
                        arrowprops=dict(arrowstyle="->", color=self.colors['neutral'], 
                                       lw=2, shrinkA=40, shrinkB=40))

        ax.set_xlim(0, 12)
        ax.set_ylim(3, 8)
        ax.axis('off')
        plt.title("NeuroDrive Autonomous Pipeline Architecture", fontsize=20, fontweight='bold', pad=20)
        
        save_path = os.path.join(self.output_dir, "pipeline_flow.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=self.colors['bg'])
        plt.close()
        print(f"  > Saved to {save_path}")

    def plot_grid_map_concept(self):
        """Illustrates the Grid Map creation process."""
        print("Generating Grid Map Illustration...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=self.colors['bg'])
        
        # 1. Raw Environment
        ax = axes[0]
        ax.set_facecolor(self.colors['bg'])
        ax.set_title("1. Environment Mapping", color=self.colors['accent1'], fontsize=14)
        # Draw lanes
        for y in [10, 30, 50, 70]:
            ax.axhline(y, color=self.colors['neutral'], alpha=0.3, ls='--')
        ax.scatter(20, 40, color=self.colors['success'], s=200, marker='>', label='Ego')
        ax.scatter(80, 40, color=self.colors['danger'], s=200, marker='X', label='Obstacle')
        ax.legend(loc='upper left', fontsize=10)
        
        # 2. Cost Calculation (Inflation)
        ax = axes[1]
        ax.set_facecolor(self.colors['bg'])
        ax.set_title("2. Risk Inflation", color=self.colors['accent2'], fontsize=14)
        # Create a mock cost map
        x, y = np.meshgrid(np.linspace(0, 100, 50), np.linspace(0, 80, 40))
        d = np.sqrt((x-80)**2 + (y-40)**2)
        cost = 100 * np.exp(-d/15)
        ax.imshow(cost, cmap='magma', extent=[0, 100, 0, 80], origin='lower')
        ax.scatter(20, 40, color=self.colors['success'], s=100, marker='>')
        
        # 3. Final Fusion
        ax = axes[2]
        ax.set_facecolor(self.colors['bg'])
        ax.set_title("3. Fused Cost Map", color=self.colors['accent3'], fontsize=14)
        # Lane costs + obstacle costs
        lane_cost = np.zeros_like(cost)
        for ly in [0, 20, 60, 80]:
            lane_cost += 50 * np.exp(-(y-ly)**2/20)
        fused = cost + lane_cost
        ax.imshow(fused, cmap='inferno', extent=[0, 100, 0, 80], origin='lower')
        
        for a in axes:
            a.set_xticks([])
            a.set_yticks([])
            for spine in a.spines.values(): spine.set_edgecolor(self.colors['neutral'])

        save_path = os.path.join(self.output_dir, "grid_map_concept.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=self.colors['bg'])
        plt.close()
        print(f"  > Saved to {save_path}")

    def plot_data_statistics(self, csv_path):
        """Generates statistical visualizations from the processed features."""
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} not found.")
            return
            
        print("Generating Data Statistics...")
        df = pd.read_csv(csv_path)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor=self.colors['bg'])
        
        # 1. Behavior Distribution
        sns.countplot(data=df, x='behavior_label', ax=axes[0, 0], palette='viridis')
        axes[0, 0].set_title("Driving Behavior Distribution", fontsize=15)
        
        # 2. Speed vs Risk
        sns.scatterplot(data=df, x='ego_speed_mps', y='computed_risk', 
                        hue='behavior_label', alpha=0.5, ax=axes[0, 1], palette='magma')
        axes[0, 1].set_title("Ego Speed vs. Computed Risk", fontsize=15)
        
        # 3. Obstacle Distance Distribution
        sns.histplot(df['obstacle_distance_m'], bins=30, kde=True, 
                     ax=axes[1, 0], color=self.colors['accent1'])
        axes[1, 0].set_title("Obstacle Distance Distribution", fontsize=15)
        
        # 4. Correlation Heatmap (Selected Features)
        cols = ['obstacle_distance_m', 'ego_speed_mps', 'computed_risk', 'ttc', 'traffic_density_veh_per_km']
        corr = df[cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1, 1], center=0)
        axes[1, 1].set_title("Feature Correlation Matrix", fontsize=15)
        
        for ax in axes.flat:
            ax.set_facecolor(self.colors['card'])
            ax.xaxis.label.set_color(self.colors['text'])
            ax.yaxis.label.set_color(self.colors['text'])
            
        plt.tight_layout(pad=4)
        save_path = os.path.join(self.output_dir, "data_statistics.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=self.colors['bg'])
        plt.close()
        print(f"  > Saved to {save_path}")

    def plot_path_planning_concept(self):
        """Visualizes how A* path planning works in the driving context."""
        print("Generating Path Planning Illustration...")
        fig, ax = plt.subplots(figsize=(12, 6), facecolor=self.colors['bg'])
        ax.set_facecolor(self.colors['bg'])
        
        # Mock grid and path
        grid_w, grid_h = 100, 3
        ax.set_xlim(-5, grid_w + 5)
        ax.set_ylim(-1, grid_h)
        
        # Draw Lanes
        for i in range(grid_h + 1):
            ax.axhline(i - 0.5, color=self.colors['neutral'], alpha=0.2)
            
        # Start and Goal
        start = (1, 5)
        goal = (1, 90)
        ax.scatter(start[1], start[0], color=self.colors['success'], s=300, marker='>', label='Start')
        ax.scatter(goal[1], goal[0], color=self.colors['accent3'], s=400, marker='*', label='Goal')
        
        # Obstacles
        obs = [(1, 40), (0, 70), (2, 20)]
        for oy, ox in obs:
            ax.scatter(ox, oy, color=self.colors['danger'], s=250, marker='X')
            # Inflation zone
            circle = patches.Circle((ox, oy), 8, color=self.colors['danger'], alpha=0.1)
            ax.add_patch(circle)
            
        # Path
        path_x = [5, 20, 30, 40, 50, 60, 70, 80, 90]
        path_y = [1, 1, 0, 0, 0, 1, 1, 1, 1]
        ax.plot(path_x, path_y, color=self.colors['accent1'], lw=4, alpha=0.8, 
                label='A* Optimized Path', zorder=5)
        
        # Search Tree nodes (Mock)
        for _ in range(100):
            rx = np.random.uniform(0, 100)
            ry = np.random.randint(0, 3)
            ax.scatter(rx, ry, color=self.colors['neutral'], s=5, alpha=0.2)

        ax.set_title("A* Trajectory Optimization in Dynamic Multi-Lane Environment", 
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', facecolor=self.colors['card'], edgecolor=self.colors['neutral'])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Lane 0', 'Lane 1', 'Lane 2'])
        
        save_path = os.path.join(self.output_dir, "path_planning_concept.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=self.colors['bg'])
        plt.close()
        print(f"  > Saved to {save_path}")

if __name__ == "__main__":
    illustrator = PipelineIllustrator()
    
    # 1. Pipeline Flow
    illustrator.plot_pipeline_flow()
    
    # 2. Grid Map Concept
    illustrator.plot_grid_map_concept()
    
    # 3. Path Planning Concept
    illustrator.plot_path_planning_concept()
    
    # 4. Data Statistics
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_root, "features", "processed_features.csv")
    illustrator.plot_data_statistics(csv_path)
    
    print("\n[SUCCESS] All pipeline illustrations have been generated in 'visualize/illustrations/'")

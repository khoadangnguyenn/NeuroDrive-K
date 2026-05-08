import heapq
import numpy as np

def calculate_heuristic(a, b):
    """
    (B) Heuristic: Manhattan distance combined with lane deviation cost.
    a, b are coordinates (y, x).
    """
    beta = 10 # Weight for lane keeping priority
    return abs(b[1] - a[1]) + beta * abs(b[0] - a[0])

def run_astar(grid, start, goal):
    """
    (A) A* Algorithm to find the safest path on the Grid Map.
    """
    rows, cols = grid.shape
    open_set = []
    # (priority, current_node)
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = {start: 0}
    
    # 3 directions: Straight, Left-diagonal, Right-diagonal.
    # Matrix system: (dy, dx). X moves forward (dx=1). Y represents lanes.
    directions = [(0, 1), (-1, 1), (1, 1)]
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1] # Return path from Start to Goal
            
        for dy, dx in directions:
            neighbor = (current[0] + dy, current[1] + dx)
            
            # Boundary check
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                # Collision check (Static obstacle)
                if grid[neighbor] >= 1e9:
                    continue
                
                # Actual movement cost (geometric)
                move_cost = np.sqrt(dy**2 + dx**2)
                
                # Penalty(Action): Slight cost for lane changes to avoid jitter
                penalty = 5 if dy != 0 else 0
                
                # Total g = accumulated cost + step cost + scaled risk + penalty
                tentative_g = g_score[current] + move_cost + (grid[neighbor] * 0.01) + penalty
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # f = g + h
                    f_score = tentative_g + calculate_heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
                    
    return None # No path found

def plan_path_with_fallback(grid, start, goal, behavior_label):
    """
    FSM Planner combining A* with fallback logic.
    
    1. If behavior is 'stop': Emergency brake.
    2. Primary A*: Search for optimal path with full constraints.
    3. Fallback A*: Relax risk costs (by 20%) to find an escape route in dangerous situations.
    """
    # Step 1: FSM Check
    if behavior_label == "stop":
        return [start], "brake", False
    
    # Step 2: Primary A*
    path = run_astar(grid, start, goal)
    if path:
        return path, "plan", False
        
    # Step 3: Fallback (Relaxed planning)
    # Reduce dynamic risk costs but keep INF for static obstacles
    relaxed_grid = np.where(grid >= 1e9, grid, grid * 0.8)
    path_relaxed = run_astar(relaxed_grid, start, goal)
    
    if path_relaxed:
        return path_relaxed, "plan", True # Fallback used
        
    return None, "brake", False # Total failure, forced to brake

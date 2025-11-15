"""
Integration module to convert frontend grid JSON data into simulation environment.

This module bridges the gap between the web UI (which generates grid JSON files)
and the simulation framework (which uses SectorEnv and Attacker objects).

KEY CHANGES:
- Frontend JSON: 0 = non-observable (protected zone), 1 = observable (open sky)
- Each attacker has its own target - no global target distribution needed
- Environment can be created without a target distribution
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.env import SectorEnv
from src.attackers import Attacker
from src.targets import PointTarget
from utils.load_grid import load_grid_payload, GridData


def _build_waypoints_for_attacker(
    start_pos: tuple[float, float],
    target_pos: tuple[float, float],
    grid_data: GridData,
    use_centroid_as_waypoint: bool,
    use_shape_centroids_as_waypoints: bool,
    additional_waypoints: list = None
) -> list[tuple[float, float]] | None:
    """
    Build ordered waypoint list for a single attacker using greedy nearest-neighbor.
    
    Strategy:
    - If multiple shapes: Use greedy nearest-neighbor to order shape centroids
      - Start from attacker's start position
      - Pick closest unvisited centroid
      - From that centroid, pick next closest unvisited centroid
      - Continue until all centroids visited
    - If single shape: Use single centroid (legacy behavior)
    - Additional waypoints are appended at the end
    
    Args:
        start_pos: Attacker start position (x, y)
        target_pos: Attacker target position (x, y)
        grid_data: GridData with shapes information
        use_centroid_as_waypoint: Whether to use centroids as waypoints
        use_shape_centroids_as_waypoints: Whether to use greedy selection for multiple shapes
        additional_waypoints: Additional waypoints to append
        
    Returns:
        Ordered list of waypoints, or None if no waypoints
    """
    waypoints_list = []
    
    if not use_centroid_as_waypoint:
        # No centroids - only use additional waypoints if provided
        if additional_waypoints is not None:
            waypoints_list.extend(additional_waypoints)
        return waypoints_list if len(waypoints_list) > 0 else None
    
    # Check if we have multiple shapes
    if len(grid_data.shapes) > 1 and use_shape_centroids_as_waypoints:
        # Multiple shapes: Use greedy nearest-neighbor selection
        # Collect all shape centroids
        centroids = []
        for shape in grid_data.shapes:
            if shape.centroid_meters is not None:
                centroids.append(tuple(shape.centroid_meters))
        
        if len(centroids) > 0:
            # Greedy nearest-neighbor ordering
            ordered_centroids = _greedy_nearest_neighbor(
                start_pos=start_pos,
                waypoint_candidates=centroids
            )
            waypoints_list.extend(ordered_centroids)
    
    elif len(grid_data.shapes) == 1:
        # Single shape: Use its centroid (legacy behavior)
        if grid_data.shapes[0].centroid_meters is not None:
            waypoints_list.append(tuple(grid_data.shapes[0].centroid_meters))
    
    elif grid_data.centroid_meters is not None:
        # Fallback to global centroid if no shapes
        waypoints_list.append(tuple(grid_data.centroid_meters))
    
    # Add any additional waypoints at the end
    if additional_waypoints is not None:
        waypoints_list.extend(additional_waypoints)
    
    # Return None if no waypoints, otherwise return the list
    return waypoints_list if len(waypoints_list) > 0 else None


def _greedy_nearest_neighbor(
    start_pos: tuple[float, float],
    waypoint_candidates: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """
    Order waypoints using greedy nearest-neighbor algorithm.
    
    Algorithm:
    1. Start at start_pos
    2. Find closest unvisited waypoint
    3. Move to that waypoint
    4. Repeat step 2-3 until all waypoints visited
    
    Args:
        start_pos: Starting position (x, y)
        waypoint_candidates: List of candidate waypoints (x, y)
        
    Returns:
        Ordered list of waypoints
    """
    if len(waypoint_candidates) == 0:
        return []
    
    # Make a copy so we can modify it
    remaining = list(waypoint_candidates)
    ordered = []
    current_pos = np.array(start_pos)
    
    # Greedy selection: always pick the closest remaining waypoint
    while remaining:
        # Calculate distances from current position to all remaining waypoints
        distances = [
            np.linalg.norm(np.array(wp) - current_pos)
            for wp in remaining
        ]
        
        # Find index of closest waypoint
        closest_idx = int(np.argmin(distances))
        
        # Add closest waypoint to ordered list
        closest_waypoint = remaining[closest_idx]
        ordered.append(closest_waypoint)
        
        # Update current position
        current_pos = np.array(closest_waypoint)
        
        # Remove from remaining
        remaining.pop(closest_idx)
    
    return ordered


def build_env_from_grid(
    grid_data: GridData,
    default_drone_speed: float = 50.0,
    default_observable: bool = False,
    use_centroid_as_waypoint: bool = True,
    use_shape_centroids_as_waypoints: bool = True,
    additional_waypoints: list = None
) -> tuple[SectorEnv, list[Attacker]]:
    """
    Build a SectorEnv and Attacker list from frontend grid data.
    
    Each attacker gets its own target distribution. The environment has no global target.
    
    Waypoint Selection Strategy:
    - If multiple shapes exist: Each attacker gets an ordered waypoint list based on
      greedy nearest-neighbor selection from shape centroids
    - If single shape: Use the centroid as waypoint (legacy behavior)
    
    Args:
        grid_data: GridData object loaded from frontend JSON
        default_drone_speed: Default speed for drones in m/s (default: 50 m/s ~ 180 km/h)
        default_observable: Start with all cells observable (True) or non-observable (False)
                          Default False means entire grid starts as non-observable (safer)
        use_centroid_as_waypoint: If True, use centroids as waypoints (default: True)
        use_shape_centroids_as_waypoints: If True and multiple shapes exist, use greedy
                                          nearest-neighbor selection from shape centroids
        additional_waypoints: Optional list of additional waypoints (x, y) tuples to add
                             These will be added AFTER shape centroids in the waypoint list
        
    Returns:
        Tuple of (SectorEnv instance, list of Attacker instances)
        
    Note:
        - Frontend matrix: 0 = non-observable (protected zone), 1 = observable (open sky)
        - SectorEnv mask: True = observable, False = non-observable
        - Coordinates are automatically handled (frontend uses row/col, simulation uses x/y)
        - Waypoint order: start -> [nearest_centroid] -> [next_nearest] -> ... -> target
    """
    
    # Get grid dimensions from matrix
    height_cells, width_cells = grid_data.matrix.shape  # Note: matrix is [rows, cols]
    cell_size = grid_data.cell_size_m
    
    # Create attackers from grid data
    attackers = []
    for attacker_data in grid_data.attackers:
        # Get positions - prioritize meter coordinates over grid coordinates
        start_pos = _get_position_meters(attacker_data.start, cell_size, height_cells)
        target_pos = _get_position_meters(attacker_data.target, cell_size, height_cells)
        
        if start_pos is not None and target_pos is not None:
            # Build waypoints list for this specific attacker
            waypoints_list = _build_waypoints_for_attacker(
                start_pos=start_pos,
                target_pos=target_pos,
                grid_data=grid_data,
                use_centroid_as_waypoint=use_centroid_as_waypoint,
                use_shape_centroids_as_waypoints=use_shape_centroids_as_waypoints,
                additional_waypoints=additional_waypoints
            )
            
            # Create individual target distribution for this attacker
            target_dist = PointTarget.at_position(x=target_pos[0], y=target_pos[1], epsilon=50.0)
            
            attacker = Attacker(
                start_position=start_pos,
                target_position=target_pos,
                target_distribution=target_dist,  # Per-attacker target
                speed=default_drone_speed,
                speed_noise=5.0,
                noise_std=0.0,  # Clean trajectories for analysis
                waypoints=waypoints_list  # Per-attacker waypoint list
            )
            attacker.position = attacker.trajectory[0]
            attackers.append(attacker)
        else:
            print(f"Warning: Skipping attacker '{attacker_data.name}' - invalid position data")
    
    # Create environment WITHOUT a global target distribution
    # Pass attackers to automatically add them
    sector_env = SectorEnv(
        width=width_cells,
        height=height_cells,
        target=None,  # No global target - each attacker has its own
        default_observable=default_observable,
        cell_size=cell_size,
        attackers=attackers  # Add attackers at initialization
    )
    
    # Apply observable mask from frontend
    # Frontend: 0 = non-observable (protected), 1 = observable (open)
    _apply_mask_to_env(sector_env, grid_data.matrix)
    
    return sector_env, attackers


def _apply_mask_to_env(sector_env: SectorEnv, matrix: np.ndarray) -> None:
    """
    Apply frontend mask to environment's observable mask.
    
    **CORRECTED INTERPRETATION:**
    Frontend matrix: 0 = non-observable (protected zone), 1 = observable (open sky)
    Environment mask: True = observable, False = non-observable
    
    The matrix uses [row, col] indexing where:
    - row increases downward (0 at top)
    - col increases rightward (0 at left)
    
    The environment uses [x, y] coordinates where:
    - x increases rightward (0 at left)
    - y increases upward (0 at bottom)
    
    So we need to convert: matrix[row, col] → observable_mask[y, x] where y = height-1-row
    """
    height, width = matrix.shape
    
    for row in range(height):
        for col in range(width):
            # Frontend: 0 = non-observable, 1 = observable
            # Environment: True = observable, False = non-observable
            # So: observable_mask = (matrix == 1)
            
            # Convert to environment coordinates
            # row=0 (top in frontend) → y=height-1 (top in env)
            # col=0 (left in frontend) → x=0 (left in env)
            x = col
            y = height - 1 - row
            
            # Set observability: 1 in matrix → True in mask, 0 → False
            sector_env.observable_mask[y, x] = (matrix[row, col] == 1)


def _get_position_meters(
    cell_pos,  # CellPosition from load_grid
    cell_size: float,
    grid_height: int
) -> tuple[float, float] | None:
    """
    Extract position in meters from CellPosition.
    
    If meters are directly provided, use them.
    Otherwise, convert from grid coordinates.
    
    Args:
        cell_pos: CellPosition with grid and/or meter coordinates
        cell_size: Size of each cell in meters
        grid_height: Total grid height in cells (needed for coordinate conversion)
        
    Returns:
        (x, y) position in meters, or None if no valid position
        
    Note:
        Frontend grid uses row/col where row=0 is at top, col=0 is at left.
        We convert to x/y where x=0 is at left, y=0 is at bottom.
    """
    # Prioritize meter coordinates if available
    if cell_pos.meters is not None:
        return tuple(cell_pos.meters)
    
    # Convert from grid coordinates
    if cell_pos.grid is not None:
        row, col = cell_pos.grid
        # Convert grid [row, col] to meters [x, y]
        # col → x (horizontal position, left to right)
        x = col * cell_size
        # row → y (vertical position, need to flip because row=0 is at top)
        y = (grid_height - 1 - row) * cell_size
        return (x, y)
    
    return None


def load_and_build_env(
    json_path: str | Path,
    default_drone_speed: float = 50.0,
    use_centroid_as_waypoint: bool = True,
    use_shape_centroids_as_waypoints: bool = True,
    additional_waypoints: list = None
) -> tuple[SectorEnv, list[Attacker], GridData]:
    """
    Convenience function to load JSON and build environment in one step.
    
    Args:
        json_path: Path to frontend-generated grid JSON file
        default_drone_speed: Default speed for drones in m/s
        use_centroid_as_waypoint: If True, use centroids as waypoints (default: True)
        use_shape_centroids_as_waypoints: If True and multiple shapes exist, use greedy
                                          nearest-neighbor selection (default: True)
        additional_waypoints: Optional list of additional waypoints (x, y) tuples
        
    Returns:
        Tuple of (SectorEnv, list of Attackers, GridData)
    """
    grid_data = load_grid_payload(json_path)
    sector_env, attackers = build_env_from_grid(
        grid_data,
        default_drone_speed=default_drone_speed,
        use_centroid_as_waypoint=use_centroid_as_waypoint,
        use_shape_centroids_as_waypoints=use_shape_centroids_as_waypoints,
        additional_waypoints=additional_waypoints
    )
    return sector_env, attackers, grid_data


if __name__ == "__main__":
    # Test with example grid
    json_path = Path(__file__).parent / "grid-12x12(2).json"
    
    if json_path.exists():
        print(f"Loading grid from: {json_path}")
        sector_env, attackers, grid_data = load_and_build_env(json_path)
        
        print(f"\n=== Grid Data ===")
        print(f"Matrix shape: {grid_data.matrix.shape}")
        print(f"Cell size: {grid_data.cell_size_m}m")
        print(f"Centroid (meters): {grid_data.centroid_meters}")
        
        print(f"\n=== Environment ===")
        print(f"Grid dimensions: {sector_env.width} × {sector_env.height} cells")
        print(f"Physical size: {sector_env.width * sector_env.cell_size:.0f} × {sector_env.height * sector_env.cell_size:.0f} meters")
        print(f"Cell size: {sector_env.cell_size}m")
        print(f"Target position: {sector_env.target}")
        
        print(f"\n=== Attackers ===")
        print(f"Number of attackers: {len(attackers)}")
        for i, attacker in enumerate(attackers):
            print(f"  Attacker {i+1}:")
            print(f"    Start: {attacker.start_position}")
            print(f"    Target: {attacker.target_position}")
            print(f"    Trajectory length: {len(attacker.trajectory)} points")
    else:
        print(f"JSON file not found: {json_path}")
        print("Please ensure a grid JSON file exists in the utils folder.")

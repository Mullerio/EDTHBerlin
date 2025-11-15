"""
Test script to verify greedy nearest-neighbor waypoint selection with multiple shapes.
"""

from pathlib import Path
import sys
import numpy as np

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.grid_integration import load_and_build_env

# Test with 12x12 grid that has 2 shapes
json_path = Path(__file__).parent / "grid-12x12(6).json"

if not json_path.exists():
    print(f"ERROR: JSON file not found: {json_path}")
    sys.exit(1)

print(f"Loading: {json_path}")
print("=" * 70)

# Load grid data first to see the shapes
from utils.load_grid import load_grid_payload
grid_data = load_grid_payload(json_path)

print(f"\n=== GRID DATA ===")
print(f"Number of shapes: {len(grid_data.shapes)}")
for i, shape in enumerate(grid_data.shapes, 1):
    print(f"\nShape {i} (ID={shape.id}):")
    print(f"  Centroid: {shape.centroid_meters}")
    print(f"  Vertices: {len(shape.vertices)}")

# Add some test attackers with different start positions
print(f"\n=== ADDING TEST ATTACKERS ===")
# Modify grid_data to add attackers at different positions
from utils.load_grid import Attacker, CellPosition

test_attackers_data = [
    {"name": "Attacker 1", "start": (1.0, 1.0), "target": (11.0, 11.0)},
    {"name": "Attacker 2", "start": (11.0, 1.0), "target": (1.0, 11.0)},
    {"name": "Attacker 3", "start": (6.0, 1.0), "target": (6.0, 11.0)},
]

grid_data.attackers = []
for data in test_attackers_data:
    attacker = Attacker(
        name=data["name"],
        start=CellPosition(grid=None, meters=np.array(data["start"])),
        target=CellPosition(grid=None, meters=np.array(data["target"]))
    )
    grid_data.attackers.append(attacker)

# Build environment with greedy waypoint selection
from utils.grid_integration import build_env_from_grid
env, attackers = build_env_from_grid(
    grid_data,
    default_drone_speed=50.0,
    use_centroid_as_waypoint=True,
    use_shape_centroids_as_waypoints=True
)

print(f"\n=== WAYPOINT ANALYSIS ===")
print(f"Number of attackers: {len(attackers)}")

# Display shape centroids for reference
shape_centroids = [tuple(shape.centroid_meters) for shape in grid_data.shapes]
print(f"\nAvailable shape centroids:")
for i, centroid in enumerate(shape_centroids, 1):
    print(f"  Shape {i}: {centroid}")

# Analyze each attacker's waypoints
for i, attacker in enumerate(attackers, 1):
    data = test_attackers_data[i-1]
    print(f"\n{'-' * 70}")
    print(f"Attacker {i}: {data['name']}")
    print(f"Start: {attacker.start_position}")
    print(f"Target: {attacker.target_position}")
    print(f"Waypoints: {attacker.waypoints}")
    
    if attacker.waypoints:
        # Calculate distances from start to each waypoint
        start = np.array(attacker.start_position)
        print(f"\nWaypoint order analysis:")
        current_pos = start
        
        for j, wp in enumerate(attacker.waypoints, 1):
            wp_array = np.array(wp)
            dist = np.linalg.norm(wp_array - current_pos)
            print(f"  {j}. Waypoint {wp} - Distance from previous: {dist:.2f}m")
            current_pos = wp_array
        
        # Final distance to target
        target = np.array(attacker.target_position)
        final_dist = np.linalg.norm(target - current_pos)
        print(f"  → Target {attacker.target_position} - Distance from last waypoint: {final_dist:.2f}m")
        
        # Verify it's greedy (each waypoint should be closest unvisited from previous position)
        print(f"\nGreedy validation:")
        remaining_centroids = list(shape_centroids)
        current = start
        for j, wp in enumerate(attacker.waypoints, 1):
            # Find which centroid this is
            distances_to_remaining = [
                np.linalg.norm(np.array(c) - current)
                for c in remaining_centroids
            ]
            closest_idx = int(np.argmin(distances_to_remaining))
            closest = remaining_centroids[closest_idx]
            
            is_closest = (tuple(wp) == closest)
            status = "✓" if is_closest else "✗"
            print(f"  Waypoint {j}: {status} {'Closest' if is_closest else 'NOT closest'} to previous position")
            
            # Update for next iteration
            current = np.array(wp)
            if tuple(wp) in remaining_centroids:
                remaining_centroids.remove(tuple(wp))

print(f"\n{'=' * 70}")
print("✅ Greedy nearest-neighbor test complete!")

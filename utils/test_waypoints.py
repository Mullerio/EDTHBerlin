"""
Test script to verify waypoint functionality with centroid.
"""

from pathlib import Path
import sys
import numpy as np

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.grid_integration import load_and_build_env

# Test with 150x150 grid
json_path = Path(__file__).parent / "grid-150x150(1).json"

if not json_path.exists():
    print(f"ERROR: JSON file not found: {json_path}")
    sys.exit(1)

print(f"Loading: {json_path}")
print("=" * 70)

# Test 1: Load with centroid as waypoint (default)
print("\n=== TEST 1: With Centroid Waypoint (default) ===")
env1, attackers1, grid_data = load_and_build_env(json_path)

print(f"Centroid: {grid_data.centroid_meters}")
print(f"Number of attackers: {len(attackers1)}")

for i, attacker in enumerate(attackers1, 1):
    print(f"\nAttacker {i}:")
    print(f"  Start: {attacker.start_position}")
    print(f"  Waypoints: {attacker.waypoints}")
    print(f"  Target: {attacker.target_position}")
    print(f"  Trajectory points: {len(attacker.trajectory)}")
    
    # Verify waypoint is in trajectory
    if attacker.waypoints:
        waypoint = attacker.waypoints[0]
        # Check if waypoint appears in trajectory (approximate match)
        traj_array = np.array(attacker.trajectory)
        distances = np.linalg.norm(traj_array - np.array(waypoint), axis=1)
        min_dist = distances.min()
        print(f"  Closest point to waypoint: {min_dist:.2f}m (should be ~0)")

# Test 2: Load without centroid waypoint
print("\n\n=== TEST 2: Without Centroid Waypoint ===")
env2, attackers2, _ = load_and_build_env(json_path, use_centroid_as_waypoint=False)

for i, attacker in enumerate(attackers2, 1):
    print(f"\nAttacker {i}:")
    print(f"  Waypoints: {attacker.waypoints}")
    print(f"  Trajectory points: {len(attacker.trajectory)}")

# Test 3: Load with additional waypoints
print("\n\n=== TEST 3: With Centroid + Additional Waypoints ===")
additional = [(5000.0, 5000.0), (10000.0, 10000.0)]
env3, attackers3, _ = load_and_build_env(
    json_path, 
    use_centroid_as_waypoint=True,
    additional_waypoints=additional
)

for i, attacker in enumerate(attackers3, 1):
    print(f"\nAttacker {i}:")
    print(f"  Waypoints: {attacker.waypoints}")
    print(f"  Expected: Centroid -> Additional waypoints")
    print(f"  Trajectory points: {len(attacker.trajectory)}")

print("\n\n" + "=" * 70)
print("✅ Waypoint tests complete!")

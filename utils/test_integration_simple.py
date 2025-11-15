"""
Simple test script to verify grid integration works correctly.
"""

from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.grid_integration import load_and_build_env

# Test with 150x150 grid
json_path = Path(__file__).parent / "grid-150x150(1).json"

if not json_path.exists():
    print(f"ERROR: JSON file not found: {json_path}")
    sys.exit(1)

print(f"Loading: {json_path}")
print("=" * 60)

# Load and build
sector_env, attackers, grid_data = load_and_build_env(json_path, default_drone_speed=50.0)

print("\n=== GRID DATA ===")
print(f"Matrix shape: {grid_data.matrix.shape} (rows × cols)")
print(f"Cell size: {grid_data.cell_size_m}m")
print(f"Non-observable cells (0): {(grid_data.matrix == 0).sum()}")
print(f"Observable cells (1): {(grid_data.matrix == 1).sum()}")

print("\n=== ENVIRONMENT ===")
print(f"Grid dimensions: {sector_env.width} × {sector_env.height} cells")
print(f"Physical size: {sector_env.width * sector_env.cell_size:.0f}m × {sector_env.height * sector_env.cell_size:.0f}m")
print(f"Target distribution: {type(sector_env.target).__name__ if sector_env.target else 'None'}")
print(f"Observable cells (True): {sector_env.observable_mask.sum()}")
print(f"Non-observable cells (False): {(~sector_env.observable_mask).sum()}")

print("\n=== ATTACKERS ===")
print(f"Number of attackers: {len(attackers)}")
print(f"Attackers in environment: {len(sector_env.atk_drones)}")

for i, attacker in enumerate(attackers, 1):
    print(f"\nAttacker {i}:")
    print(f"  Start: {attacker.start_position}")
    print(f"  Target: {attacker.target_position}")
    print(f"  Speed: {attacker.speed} m/s")
    print(f"  Trajectory points: {len(attacker.trajectory)}")
    
    # Calculate distance
    import numpy as np
    start = np.array(attacker.start_position)
    target = np.array(attacker.target_position)
    distance = np.linalg.norm(target - start)
    flight_time = len(attacker.trajectory)
    
    print(f"  Distance: {distance:.0f}m ({distance/1000:.2f}km)")
    print(f"  Flight time: {flight_time}s ({flight_time/60:.1f} min)")

print("\n=== VERIFICATION ===")
# Verify mask interpretation
print("Checking mask interpretation...")
# Sample a few cells from the matrix and check they match the environment mask
test_points = [(0, 0), (10, 10), (50, 50), (100, 100)]
for row, col in test_points:
    if row < grid_data.matrix.shape[0] and col < grid_data.matrix.shape[1]:
        frontend_value = grid_data.matrix[row, col]
        # Convert to env coordinates
        x = col
        y = grid_data.matrix.shape[0] - 1 - row
        env_value = sector_env.observable_mask[y, x]
        
        # Frontend: 0 = non-obs, 1 = obs
        # Environment: False = non-obs, True = obs
        expected = (frontend_value == 1)
        match = "✓" if env_value == expected else "✗"
        print(f"  [{row},{col}] frontend={frontend_value}, env={env_value}, expected={expected} {match}")

print("\n✅ Integration test complete!")

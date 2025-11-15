"""
Test to verify that detection statistics are independent of grid resolution.

This demonstrates that:
1. Observable/non-observable boundaries depend on grid (as intended - for UI selection)
2. Detection probabilities along trajectories are continuous and grid-independent
3. Changing cell_size doesn't affect detection statistics (only boundary granularity)
"""

import sys
import os

# Add src and configs directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(parent_dir, 'src')
configs_dir = os.path.join(parent_dir, 'configs')
sys.path.insert(0, src_dir)
sys.path.insert(0, configs_dir)

import numpy as np
from env import SectorEnv
from attackers import Attacker
from detectors import Detector, Rect_Detectors
import detector_configs
from targets import PointTarget


def test_grid_independence():
    """
    Test that detection statistics don't change when grid resolution changes.
    """
    print("=" * 70)
    print("GRID INDEPENDENCE TEST")
    print("=" * 70)
    print("\nThis test creates identical physical scenarios with different grid resolutions")
    print("and verifies that detection probabilities remain the same.\n")
    
    # Physical scenario parameters (in meters)
    physical_width = 400.0  # meters
    physical_height = 400.0  # meters
    
    # Non-observable region in physical coordinates (meters)
    nonobs_x0_phys = 100.0
    nonobs_y0_phys = 100.0
    nonobs_width_phys = 200.0
    nonobs_height_phys = 200.0
    
    # Detector positions in physical coordinates (meters)
    detector_positions_phys = [
        (150.0, 150.0),
        (200.0, 200.0),
        (250.0, 150.0),
    ]
    
    # Attacker trajectory in physical coordinates (meters)
    attacker_start_phys = (50.0, 200.0)
    attacker_target_phys = (350.0, 200.0)
    attacker_speed = 2.0  # meters per second
    
    # Target (for visualization only, not used in detection)
    target = PointTarget.at_position(x=350.0, y=200.0, epsilon=0.5)
    
    # Test with different grid resolutions
    cell_sizes = [1.0, 2.0, 5.0, 10.0]  # meters per cell
    results = []
    
    for cell_size in cell_sizes:
        print(f"\n{'='*70}")
        print(f"Testing with cell_size = {cell_size} meters per cell")
        print(f"{'='*70}")
        
        # Calculate grid dimensions (number of cells)
        width_cells = int(np.ceil(physical_width / cell_size))
        height_cells = int(np.ceil(physical_height / cell_size))
        
        print(f"Grid: {width_cells} x {height_cells} cells")
        print(f"Physical area: {physical_width} x {physical_height} meters")
        
        # Create environment
        env = SectorEnv(width_cells, height_cells, target, 
                       default_observable=True, cell_size=cell_size)
        
        # Set non-observable region (convert physical coords to grid cells)
        nonobs_x0_cells = int(nonobs_x0_phys / cell_size)
        nonobs_y0_cells = int(nonobs_y0_phys / cell_size)
        nonobs_w_cells = int(nonobs_width_phys / cell_size)
        nonobs_h_cells = int(nonobs_height_phys / cell_size)
        
        env.set_rectangular_sector(
            x0=nonobs_x0_cells,
            y0=nonobs_y0_cells,
            w=nonobs_w_cells,
            h=nonobs_h_cells,
            observable=False
        )
        
        # Add detectors at physical positions (meters)
        for pos_phys in detector_positions_phys:
            det = Detector(type=detector_configs.DetectorType.RADAR, position=pos_phys)
            env.add_detector(det)
        
        # Create attacker with trajectory in physical coordinates (meters)
        attacker = Attacker(
            start_position=attacker_start_phys,
            target_position=attacker_target_phys,
            speed=attacker_speed,
            noise_std=0.0  # No noise for deterministic test
        )
        attacker.position = attacker.trajectory[0]
        env.add_atk_drone(attacker)
        
        # Analyze trajectory
        analysis = env.analyze_trajectory(
            attacker.trajectory,
            time_per_step=1.0,
            only_nonobservable=True
        )
        
        # Store results
        result = {
            'cell_size': cell_size,
            'grid_dims': (width_cells, height_cells),
            'trajectory_length': len(attacker.trajectory),
            'time_in_nonobs': analysis['time_in_nonobservable'],
            'cumulative_detection': analysis['cumulative_detection_prob'],
            'avg_detection_per_sec': analysis['avg_detection_per_second'],
            'detection_10s': analysis['detection_rate_per_interval']['10s'],
        }
        results.append(result)
        
        # Print results
        print(f"\nResults:")
        print(f"  Trajectory points: {result['trajectory_length']}")
        print(f"  Time in non-observable: {result['time_in_nonobs']:.1f}s")
        print(f"  Cumulative detection prob: {result['cumulative_detection']:.4f}")
        print(f"  Detection per second: {result['avg_detection_per_sec']:.6f}")
        print(f"  Detection per 10s: {result['detection_10s']:.4f}")
    
    # Verify consistency
    print(f"\n{'='*70}")
    print("VERIFICATION: Comparing results across grid resolutions")
    print(f"{'='*70}\n")
    
    # Check trajectory length consistency (should be same - determined by speed and distance)
    traj_lengths = [r['trajectory_length'] for r in results]
    print(f"Trajectory lengths: {traj_lengths}")
    if len(set(traj_lengths)) == 1:
        print("✓ PASS: Trajectory length is consistent across all grid resolutions")
    else:
        print("✗ FAIL: Trajectory length varies with grid resolution!")
    
    # Check detection statistics consistency
    ref = results[0]
    tolerance = 0.02  # 2% tolerance for floating point / boundary effects
    
    all_consistent = True
    for r in results[1:]:
        # Time in non-observable may vary slightly due to grid boundary discretization
        time_diff = abs(r['time_in_nonobs'] - ref['time_in_nonobs'])
        time_rel_diff = time_diff / max(ref['time_in_nonobs'], 1.0)
        
        # Detection probabilities should be very consistent (continuous computation)
        det_diff = abs(r['cumulative_detection'] - ref['cumulative_detection'])
        det_rel_diff = det_diff / max(ref['cumulative_detection'], 1e-6)
        
        print(f"\nComparing cell_size={r['cell_size']} vs {ref['cell_size']}:")
        print(f"  Time in non-obs difference: {time_diff:.2f}s ({time_rel_diff*100:.1f}%)")
        print(f"  Detection prob difference: {det_diff:.6f} ({det_rel_diff*100:.2f}%)")
        
        if time_rel_diff > tolerance:
            print(f"  ⚠ Time difference exceeds {tolerance*100}% (expected due to grid boundary)")
        else:
            print(f"  ✓ Time difference within {tolerance*100}%")
        
        if det_rel_diff > tolerance:
            print(f"  ✗ Detection probability difference exceeds {tolerance*100}%!")
            all_consistent = False
        else:
            print(f"  ✓ Detection probability within {tolerance*100}%")
    
    print(f"\n{'='*70}")
    if all_consistent:
        print("✓ SUCCESS: Detection probabilities are grid-independent!")
        print("\nThe detection statistics are computed continuously and do not depend")
        print("on grid resolution. Only the observable boundary (from the mask) depends")
        print("on grid granularity, which is expected for UI-based region selection.")
    else:
        print("✗ FAILURE: Detection probabilities vary with grid resolution")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    results = test_grid_independence()

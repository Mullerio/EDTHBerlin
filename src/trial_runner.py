"""
Trial runner for detector optimization experiments.

This module provides a function to run a single trial experiment:
1. Load environment from JSON
2. Analyze trajectories WITHOUT detectors
3. Optimize detector placement
4. Analyze trajectories WITH detectors
5. Save results to CSV files

Results are saved for both include_observable_in_stats modes (True/False).
"""

import csv
import sys
from pathlib import Path
from typing import Literal

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - no display, only save
import matplotlib.pyplot as plt

import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.grid_integration import load_and_build_env
from src.detectors import optimize_detector_positions
from configs import detector_configs


def run_single_trial(
    json_path: str | Path,
    n_detectors: int,
    detector_type: detector_configs.DetectorType,
    sliding_window_sizes: list[int],
    optimization_method: Literal['greedy', 'refined', 'greedy+refine'],
    output_dir: str | Path,
    both_observable_modes: bool = True,
    default_drone_speed: float = 50.0,
    waypoint_mode: Literal['shapes', 'grid_center', 'none'] = 'shapes',
    trajectory_noise_std: float = 0.0,
    use_swarm_position: bool = False,
    swarm_spread: float = 0.0,
    attackers_per_swarm: int = 4
) -> dict:
    """
    Run a single trial experiment with detector optimization.
    
    This function:
    1. Loads environment from JSON
    2. Analyzes trajectories WITHOUT detectors (baseline)
    3. Optimizes detector positions using specified method
    4. Analyzes trajectories WITH detectors
    5. Saves results to CSV files
    
    Parameters
    ----------
    json_path : str or Path
        Path to the JSON file containing the grid configuration
    n_detectors : int
        Number of detectors to place
    detector_type : DetectorType
        Type of detector to use (VISUAL, RADAR, ACOUSTIC, BASE)
    sliding_window_sizes : list[int]
        List of sliding window sizes in seconds (e.g., [5, 10, 15, 30, 60])
    optimization_method : {'greedy', 'refined', 'greedy+refine'}
        Optimization method for detector placement:
        - 'greedy': Fast greedy algorithm
        - 'refined': Greedy + local optimization (slower but better)
        - 'greedy+refine': Alias for 'refined' (default)
    output_dir : str or Path
        Directory to save CSV output files
    both_observable_modes : bool, default=True
        If True, analyze with both include_observable_in_stats=True and False
        If False, only analyze with include_observable_in_stats=False
    default_drone_speed : float, default=50.0
        Default speed for attacker drones in m/s
    waypoint_mode : {'shapes', 'grid_center', 'none'}, default='shapes'
        Waypoint strategy for drone trajectories:
        - 'shapes': Use shape centroids with greedy nearest-neighbor (default)
        - 'grid_center': All drones fly to grid center, then to target
        - 'none': Direct straight line from start to target (no waypoints)
    trajectory_noise_std : float, default=0.0
        Standard deviation of Gaussian noise to add to trajectory waypoints (in meters)
        If 0.0, no noise is added. Recommended values: 5.0-50.0 for realistic deviation.
        Noise is ALWAYS applied during trajectory generation (not post-processing) to
        ensure proper timing and distance calculations:
        - When use_swarm_position=True: Applied by swarm class during generation
        - When use_swarm_position=False: Trajectories regenerated with noise_std parameter
    use_swarm_position : bool, default=False
        If True, treat attacker positions as swarm center positions instead of individual
        attacker positions. Each attacker's actual position is sampled around the swarm
        center based on swarm_spread parameter. trajectory_noise_std is used for swarm
        trajectory noise.
    swarm_spread : float, default=0.0
        When use_swarm_position=True, this is the spread radius (in meters) for sampling
        individual attacker positions around the swarm center. Each attacker's start
        position is uniformly sampled within [-spread, +spread] in each dimension.
    attackers_per_swarm : int, default=1
        When use_swarm_position=True, this is the number of attackers to generate per
        swarm position. For example, if the JSON has 4 attacker positions and 
        attackers_per_swarm=3, this will generate 12 total attackers (3 per position).
        Each attacker in a swarm will have different start positions (based on spread)
        and different trajectory noise. Ignored when use_swarm_position=False.
        
    Returns
    -------
    dict
        Dictionary containing trial results and metadata:
        - 'json_path': Path to input JSON
        - 'n_detectors': Number of detectors placed
        - 'detector_type': Type of detector used
        - 'optimization_method': Optimization method used
        - 'detector_positions': List of detector positions
        - 'detector_metrics': Metrics from detector optimization
        - 'csv_files': List of generated CSV file paths
        
    Output CSV Files
    ----------------
    The function generates the following CSV files in output_dir:
    
    1. baseline_exclude_observable.csv:
       - Trajectory analysis WITHOUT detectors, excluding observable regions
       
    2. baseline_include_observable.csv (if both_observable_modes=True):
       - Trajectory analysis WITHOUT detectors, including observable regions
       
    3. with_detectors_exclude_observable.csv:
       - Trajectory analysis WITH detectors, excluding observable regions
       
    4. with_detectors_include_observable.csv (if both_observable_modes=True):
       - Trajectory analysis WITH detectors, including observable regions
    
    Each CSV contains:
    - trajectory_id: Index of the attacker/trajectory
    - cumulative_detection_prob: Final cumulative detection probability
    - avg_detection_per_second: Average instantaneous detection probability per second
    - time_in_observable: Time spent in observable regions (seconds)
    - time_in_nonobservable: Time spent in non-observable regions (seconds)
    - sliding_window_{size}s_min: Minimum detection probability in {size}s windows
    - sliding_window_{size}s_max: Maximum detection probability in {size}s windows
    - sliding_window_{size}s_mean: Mean detection probability in {size}s windows
    
    Plus a summary row with 'trajectory_id'='MEAN' containing means across all trajectories.
    
    Example
    -------
    >>> from pathlib import Path
    >>> from configs.detector_configs import DetectorType
    >>> 
    >>> results = run_single_trial(
    ...     json_path="grid-150x150(3).json",
    ...     n_detectors=20,
    ...     detector_type=DetectorType.VISUAL,
    ...     sliding_window_sizes=[5, 10, 15, 30, 60],
    ...     optimization_method='greedy+refine',
    ...     output_dir="results/trial_001",
    ...     both_observable_modes=True
    ... )
    >>> 
    >>> print(f"Placed {len(results['detector_positions'])} detectors")
    >>> print(f"Generated {len(results['csv_files'])} CSV files")
    """
    
    # Convert paths to Path objects
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load JSON and build environment
    # ========================================================================
    print(f"Loading environment from {json_path}...")
    
    # Configure waypoint settings based on mode
    if waypoint_mode == 'shapes':
        use_centroid_as_waypoint = True
        use_shape_centroids_as_waypoints = True
    elif waypoint_mode == 'grid_center':
        use_centroid_as_waypoint = False  # Don't use shape centroids
        use_shape_centroids_as_waypoints = False
    else:  # waypoint_mode == 'none'
        use_centroid_as_waypoint = False
        use_shape_centroids_as_waypoints = False
    
    sector_env, attackers, grid_data = load_and_build_env(
        json_path,
        default_drone_speed=default_drone_speed,
        use_centroid_as_waypoint=use_centroid_as_waypoint,
        use_shape_centroids_as_waypoints=use_shape_centroids_as_waypoints
    )
    
    # Apply trajectory noise or swarm positioning
    if use_swarm_position:
        from src.attackers import AttackerSwarm
        
        print(f"Using swarm positions (spread={swarm_spread}m, trajectory_noise={trajectory_noise_std}m, {attackers_per_swarm} attackers/swarm)...")
        
        # Group attackers by their original start positions to create swarms
        swarm_dict = {}
        for attacker in attackers:
            start_key = attacker.start_position
            if start_key not in swarm_dict:
                swarm_dict[start_key] = []
            swarm_dict[start_key].append(attacker)
        
        # Generate new attackers using swarm positions
        new_attackers = []
        for swarm_center, attacker_group in swarm_dict.items():
            # Collect waypoints and targets from original attackers
            waypoints_to_use = None
            if len(attacker_group) > 0 and attacker_group[0].waypoints:
                waypoints_to_use = attacker_group[0].waypoints
            
            # Create a swarm for each original attacker in this group
            for attacker in attacker_group:
                swarm = AttackerSwarm(
                    start_position=swarm_center,
                    target_positions=[attacker.target_position],
                    number_of_attackers=attackers_per_swarm,  # Generate multiple attackers per swarm
                    spread=swarm_spread,
                    noise_std=trajectory_noise_std,  # Use trajectory_noise_std for swarm noise
                    waypoints=waypoints_to_use
                )
                # Generate attackers from swarm (with spread and noise applied)
                swarm_attackers = swarm.generate_swarm(
                    steps=len(attacker.trajectory),
                    speed=default_drone_speed
                )
                new_attackers.extend(swarm_attackers)
        
        attackers = new_attackers
        original_count = len(swarm_dict) * len(next(iter(swarm_dict.values())))
        print(f"   Generated {len(attackers)} attackers from {original_count} base positions ({attackers_per_swarm} per swarm)")
        
        # IMPORTANT: Update the environment's atk_drones list to reflect the new swarm attackers
        # This ensures visualizations show the actual swarm members, not the old swarm centers
        sector_env.atk_drones = attackers
        print(f"   Updated environment to track {len(attackers)} swarm attackers")
    
    elif trajectory_noise_std > 0.0:
        # Not using swarm, but want noise during trajectory generation
        print(f"Regenerating trajectories with noise (std={trajectory_noise_std}m)...")
        for i, attacker in enumerate(attackers):
            # Update attacker's noise_std and regenerate trajectory
            attacker.noise_std = trajectory_noise_std
            attacker.trajectory = attacker.generate_trajectory()
        print(f"   Regenerated {len(attackers)} trajectories with noise applied during generation")
    
    # Override waypoints based on mode
    if waypoint_mode == 'grid_center':
        # Calculate grid center
        grid_center_x = (sector_env.width * sector_env.cell_size) / 2
        grid_center_y = (sector_env.height * sector_env.cell_size) / 2
        grid_center = (grid_center_x, grid_center_y)
        
        print(f"Overriding waypoints to grid center: {grid_center}")
        
        # Override all attacker waypoints to use grid center
        for i, attacker in enumerate(attackers):
            old_waypoints = attacker.waypoints
            attacker.waypoints = [grid_center]
            # Regenerate trajectory with new waypoint - MUST assign back to trajectory!
            attacker.trajectory = attacker.generate_trajectory()
            print(f"   Attacker {i}: waypoints changed from {old_waypoints} to {attacker.waypoints}")
            print(f"   Attacker {i}: trajectory regenerated with {len(attacker.trajectory)} points")
    
    elif waypoint_mode == 'none':
        # Remove all waypoints - direct line to target
        print(f"Removing all waypoints for direct trajectories")
        for i, attacker in enumerate(attackers):
            old_waypoints = attacker.waypoints
            attacker.waypoints = None
            # Regenerate trajectory without waypoints - MUST assign back to trajectory!
            attacker.trajectory = attacker.generate_trajectory()
            print(f"   Attacker {i}: waypoints changed from {old_waypoints} to None")
    
    print(f"✅ Environment loaded:")
    print(f"   Grid: {sector_env.width}×{sector_env.height} cells @ {sector_env.cell_size}m/cell")
    print(f"   Shapes: {len(grid_data.shapes)}")
    print(f"   Attackers: {len(attackers)}")
    print(f"   Waypoint mode: {waypoint_mode}")
    if use_swarm_position:
        print(f"   Swarm mode: ENABLED (spread={swarm_spread}m, {attackers_per_swarm} attackers/swarm, trajectory_noise={trajectory_noise_std}m)")
    else:
        print(f"   Trajectory noise: {trajectory_noise_std}m std")
    
    # ========================================================================
    # STEP 2: Analyze trajectories WITHOUT detectors (baseline)
    # ========================================================================
    print(f"\n{'='*70}")
    print("STEP 2: Baseline Analysis (WITHOUT Detectors)")
    print(f"{'='*70}")
    
    # Determine which modes to run
    observable_modes = [False, True] if both_observable_modes else [False]
    
    # Create plots subdirectory
    plots_dir = output_dir / "plots_baseline"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    for include_observable in observable_modes:
        mode_str = "INCLUDE" if include_observable else "EXCLUDE"
        print(f"\nAnalyzing trajectories ({mode_str} observable regions)...")
        
        # Collect results for all trajectories
        trajectory_results = []
        
        for i, attacker in enumerate(attackers):
            analysis = sector_env.analyze_trajectory(
                attacker.trajectory,
                time_per_step=1.0,
                only_nonobservable=True,
                include_observable_in_stats=include_observable
            )
            
            # Save visualization plot for this trajectory
            plot_filename = f"trajectory_{i}_{'include' if include_observable else 'exclude'}_observable.png"
            plot_path = plots_dir / plot_filename
            
            # Create and save visualization (non-interactive, no display)
            sector_env.visualize_trajectory_analysis(
                attacker,
                figsize=(14, 6),
                include_observable_in_stats=include_observable
            )
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close('all')  # Close all figures to free memory
            
            # Extract data
            result = {
                'trajectory_id': i,
                'cumulative_detection_prob': analysis['cumulative_detection_prob'],
                'avg_detection_per_second': analysis['avg_detection_per_second'],
                'time_in_observable': analysis['time_in_observable'],
                'time_in_nonobservable': analysis['time_in_nonobservable']
            }
            
            # Extract sliding window statistics
            sw_summary = analysis['sliding_window_stats']['summary']
            for size in sliding_window_sizes:
                size_key = f'{size}s'
                if size_key in sw_summary:
                    result[f'sliding_window_{size}s_min'] = sw_summary[size_key]['min']
                    result[f'sliding_window_{size}s_max'] = sw_summary[size_key]['max']
                    result[f'sliding_window_{size}s_mean'] = sw_summary[size_key]['mean']
                else:
                    result[f'sliding_window_{size}s_min'] = np.nan
                    result[f'sliding_window_{size}s_max'] = np.nan
                    result[f'sliding_window_{size}s_mean'] = np.nan
            
            trajectory_results.append(result)
        
        # Create combined visualization showing all trajectories together
        if len(attackers) > 1:
            print(f"Creating combined visualization with all {len(attackers)} trajectories...")
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            sector_env.visualize(ax=ax, show=False, show_sectors=True)
            
            # Plot all trajectories
            for i, attacker in enumerate(attackers):
                traj_arr = np.array(attacker.trajectory)
                ax.plot(traj_arr[:, 0], traj_arr[:, 1], '-', linewidth=1.5, alpha=0.7, label=f'Traj {i}')
                ax.plot(traj_arr[0, 0], traj_arr[0, 1], 'go', markersize=8, alpha=0.6)
                ax.plot(traj_arr[-1, 0], traj_arr[-1, 1], 'rx', markersize=10, alpha=0.6)
            
            ax.set_title(f'All {len(attackers)} Trajectories - Baseline')
            # Only show legend if not too many trajectories
            if len(attackers) <= 20:
                ax.legend(loc='upper right', fontsize=8, ncol=2)
            
            # Save combined plot
            combined_plot_filename = f"all_trajectories_{'include' if include_observable else 'exclude'}_observable.png"
            combined_plot_path = plots_dir / combined_plot_filename
            plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
            plt.close('all')
            print(f"   Saved combined plot to {combined_plot_filename}")
        
        # Calculate means across all trajectories
        mean_result = {'trajectory_id': 'MEAN'}
        
        # Get all numeric keys
        numeric_keys = [k for k in trajectory_results[0].keys() if k != 'trajectory_id']
        
        for key in numeric_keys:
            values = [r[key] for r in trajectory_results if not np.isnan(r[key])]
            mean_result[key] = np.mean(values) if values else np.nan
        
        trajectory_results.append(mean_result)
        
        # Save to CSV
        csv_filename = f"baseline_{'include' if include_observable else 'exclude'}_observable.csv"
        csv_path = output_dir / csv_filename
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['trajectory_id'] + numeric_keys
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in trajectory_results:
                writer.writerow(result)
        
        print(f"✅ Saved baseline results to {csv_path}")
        print(f"   Trajectories analyzed: {len(attackers)}")
        print(f"   Mean cumulative detection: {mean_result['cumulative_detection_prob']:.2%}")
        print(f"   Mean detection per second: {mean_result['avg_detection_per_second']:.4%}")
    
    # ========================================================================
    # STEP 3: Optimize detector positions
    # ========================================================================
    print(f"\n{'='*70}")
    print("STEP 3: Detector Optimization")
    print(f"{'='*70}")
    
    # Get non-observable region mask (0 = non-observable, 1 = observable)
    # optimize_detector_positions expects 0 = area to cover
    area_grid = sector_env.observable_mask.astype(int)
    
    print(f"Optimizing {n_detectors} {detector_type.value} detectors using '{optimization_method}' method...")
    
    detector_positions, detector_metrics = optimize_detector_positions(
        area_grid=area_grid,
        n_detectors=n_detectors,
        detector_config=None,  # Will use detector_type
        cell_size=sector_env.cell_size,
        method=optimization_method,
        detector_type=detector_type
    )
    
    print(f"✅ Detector optimization complete:")
    print(f"   Positions found: {len(detector_positions)}")
    print(f"   Mean coverage: {detector_metrics.get('mean_coverage', 0.0):.2%}")
    print(f"   Min coverage: {detector_metrics.get('min_coverage', 0.0):.2%}")
    print(f"   Fraction covered: {detector_metrics.get('fraction_covered', 0.0):.2%}")
    
    # Add detectors to environment
    from src.detectors import Detector
    
    for pos in detector_positions:
        detector = Detector(type=detector_type, position=pos)
        sector_env.add_detector(detector)
    
    print(f"✅ Added {len(sector_env.detectors)} detectors to environment")
    
    # ========================================================================
    # STEP 4: Analyze trajectories WITH detectors
    # ========================================================================
    print(f"\n{'='*70}")
    print("STEP 4: Analysis WITH Detectors")
    print(f"{'='*70}")
    
    csv_files = []
    
    # Create plots subdirectory for with-detectors analysis
    plots_dir_detectors = output_dir / "plots_with_detectors"
    plots_dir_detectors.mkdir(parents=True, exist_ok=True)
    
    for include_observable in observable_modes:
        mode_str = "INCLUDE" if include_observable else "EXCLUDE"
        print(f"\nAnalyzing trajectories ({mode_str} observable regions)...")
        
        # Collect results for all trajectories
        trajectory_results = []
        
        for i, attacker in enumerate(attackers):
            analysis = sector_env.analyze_trajectory(
                attacker.trajectory,
                time_per_step=1.0,
                only_nonobservable=True,
                include_observable_in_stats=include_observable
            )
            
            # Save visualization plot for this trajectory
            plot_filename = f"trajectory_{i}_{'include' if include_observable else 'exclude'}_observable.png"
            plot_path = plots_dir_detectors / plot_filename
            
            # Create and save visualization (non-interactive, no display)
            sector_env.visualize_trajectory_analysis(
                attacker,
                figsize=(14, 6),
                include_observable_in_stats=include_observable
            )
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close('all')  # Close all figures to free memory
            
            # Extract data
            result = {
                'trajectory_id': i,
                'cumulative_detection_prob': analysis['cumulative_detection_prob'],
                'avg_detection_per_second': analysis['avg_detection_per_second'],
                'time_in_observable': analysis['time_in_observable'],
                'time_in_nonobservable': analysis['time_in_nonobservable']
            }
            
            # Extract sliding window statistics
            sw_summary = analysis['sliding_window_stats']['summary']
            for size in sliding_window_sizes:
                size_key = f'{size}s'
                if size_key in sw_summary:
                    result[f'sliding_window_{size}s_min'] = sw_summary[size_key]['min']
                    result[f'sliding_window_{size}s_max'] = sw_summary[size_key]['max']
                    result[f'sliding_window_{size}s_mean'] = sw_summary[size_key]['mean']
                else:
                    result[f'sliding_window_{size}s_min'] = np.nan
                    result[f'sliding_window_{size}s_max'] = np.nan
                    result[f'sliding_window_{size}s_mean'] = np.nan
            
            trajectory_results.append(result)
        
        # Create combined visualization showing all trajectories together
        if len(attackers) > 1:
            print(f"Creating combined visualization with all {len(attackers)} trajectories...")
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            sector_env.visualize(ax=ax, show=False, show_sectors=True)
            
            # Plot all trajectories
            for i, attacker in enumerate(attackers):
                traj_arr = np.array(attacker.trajectory)
                ax.plot(traj_arr[:, 0], traj_arr[:, 1], '-', linewidth=1.5, alpha=0.7, label=f'Traj {i}')
                ax.plot(traj_arr[0, 0], traj_arr[0, 1], 'go', markersize=8, alpha=0.6)
                ax.plot(traj_arr[-1, 0], traj_arr[-1, 1], 'rx', markersize=10, alpha=0.6)
            
            ax.set_title(f'All {len(attackers)} Trajectories - With Detectors')
            # Only show legend if not too many trajectories
            if len(attackers) <= 20:
                ax.legend(loc='upper right', fontsize=8, ncol=2)
            
            # Save combined plot
            combined_plot_filename = f"all_trajectories_{'include' if include_observable else 'exclude'}_observable.png"
            combined_plot_path = plots_dir_detectors / combined_plot_filename
            plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
            plt.close('all')
            print(f"   Saved combined plot to {combined_plot_filename}")
        
        # Calculate means across all trajectories
        mean_result = {'trajectory_id': 'MEAN'}
        
        # Get all numeric keys
        numeric_keys = [k for k in trajectory_results[0].keys() if k != 'trajectory_id']
        
        for key in numeric_keys:
            values = [r[key] for r in trajectory_results if not np.isnan(r[key])]
            mean_result[key] = np.mean(values) if values else np.nan
        
        trajectory_results.append(mean_result)
        
        # Save to CSV
        csv_filename = f"with_detectors_{'include' if include_observable else 'exclude'}_observable.csv"
        csv_path = output_dir / csv_filename
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['trajectory_id'] + numeric_keys
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in trajectory_results:
                writer.writerow(result)
        
        csv_files.append(str(csv_path))
        
        print(f"✅ Saved detector results to {csv_path}")
        print(f"   Trajectories analyzed: {len(attackers)}")
        print(f"   Mean cumulative detection: {mean_result['cumulative_detection_prob']:.2%}")
        print(f"   Mean detection per second: {mean_result['avg_detection_per_second']:.4%}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("TRIAL COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"CSV files generated: {len(csv_files) + len(observable_modes)}")
    print(f"Baseline plots saved to: {plots_dir}")
    print(f"Detector plots saved to: {plots_dir_detectors}")
    individual_plots = len(attackers) * len(observable_modes) * 2
    combined_plots = len(observable_modes) * 2 if len(attackers) > 1 else 0
    print(f"Total plots generated: {individual_plots + combined_plots} ({individual_plots} individual + {combined_plots} combined)")
    
    return {
        'json_path': str(json_path),
        'n_detectors': n_detectors,
        'detector_type': detector_type.value,
        'optimization_method': optimization_method,
        'detector_positions': detector_positions,
        'detector_metrics': detector_metrics,
        'csv_files': csv_files,
        'output_dir': str(output_dir),
        'plots_baseline_dir': str(plots_dir),
        'plots_detectors_dir': str(plots_dir_detectors)
    }


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    
    # Get the project root directory (parent of src/)
    project_root = Path(__file__).parent.parent
    
    # Example trial with shape centroids waypoints (default)
    results = run_single_trial(
        json_path=project_root / "utils" / "basic_notreal.json",
        n_detectors=10,
        detector_type=detector_configs.DetectorType.VISUAL,
        sliding_window_sizes=[5, 10, 15, 30, 60],
        optimization_method='greedy+refine',
        output_dir=project_root / "results" / "trial_example",
        both_observable_modes=True,
        waypoint_mode='grid_center',  # Options: 'shapes', 'grid_center', 'none'
        trajectory_noise_std=40.0,  # Trajectory noise (meters) - used for both normal and swarm mode
        use_swarm_position=True,  # Use swarm positioning
        swarm_spread=250.0  # Spread radius (meters) when use_swarm_position=True
    )
    
    print(f"\n\nTrial metadata:")
    for key, value in results.items():
        if key != 'detector_positions':  # Skip printing all positions
            print(f"  {key}: {value}")
    
    # ========================================================================
    # Configuration Examples:
    # ========================================================================
    
    # Waypoint modes:
    # waypoint_mode='shapes'       # Use shape centroids (default)
    # waypoint_mode='grid_center'  # All drones fly to grid center first
    # waypoint_mode='none'         # Straight line from start to target
    
    # Trajectory noise (all values in meters):
    # trajectory_noise_std=20.0    # 20m standard deviation
    #                              # ALWAYS applied during trajectory generation (not post-processing)
    #                              # Ensures proper timing and distance calculations
    
    # Swarm positioning (treats attacker positions as swarm centers):
    # use_swarm_position=True      # Enable swarm mode
    # swarm_spread=50.0            # Individual attackers spread 50m around center
    #                              # trajectory_noise_std applies to each swarm attacker

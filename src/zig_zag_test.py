"""Quick test to verify zig-zag is visible in trial_runner output"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.trial_runner import run_single_trial
from configs import detector_configs

project_root = Path(__file__).parent

print("="*70)
print("RUNNING SINGLE TRIAL WITH ZIG-ZAG ENABLED")
print("="*70)

results = run_single_trial(
    json_path=project_root / "utils" / "presentation.json",
    n_detectors=10,
    detector_type=detector_configs.DetectorType.VISUAL,
    sliding_window_sizes=[5, 10, 15],
    optimization_method='greedy',
    output_dir=project_root / "results" / "test_zigzag",
    waypoint_mode='grid_center',
    trajectory_noise_std=0.0,  # No noise so we see pure zig-zag
    use_swarm_position=True,
    swarm_spread=700.0,
    attackers_per_swarm=3,  # Just 3 per swarm for quick run
    save_plots=True,
    minimal_mode=True,  # Skip baseline
    use_dynamic_trajectory=True,  # ENABLE ZIG-ZAG
    trajectory_aggressiveness=2.0
)

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
print(f"Check plots in: {results['plots_detectors_dir']}")
print("Look for obvious zig-zag pattern in the trajectory plots!")

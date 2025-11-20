"""
Quick visual test: Generate and plot a single trajectory with zig-zag + noise
to verify the pattern is visible before running a full sweep.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import matplotlib.pyplot as plt
import numpy as np
from src.attackers import Attacker

# Test parameters
start = (6000, 6000)
mid = (16000, 16000)
target = (28000, 20000)

# Generate three trajectories for comparison:
# 1. Linear (no zig-zag, no noise)
# 2. Zig-zag only (no noise)
# 3. Zig-zag with noise

print("Generating trajectories...")

# 1. Linear
a_linear = Attacker(
    start_position=start,
    waypoints=[mid],
    target_position=target,
    use_dynamic_trajectory=False,
    steps=200,
    noise_std=0.0
)

# 2. Zig-zag only (large amplitude for visibility)
a_zigzag = Attacker(
    start_position=start,
    waypoints=[mid],
    target_position=target,
    use_dynamic_trajectory=True,
    trajectory_aggressiveness=2.0,
    steps=600,
    noise_std=0.0
)
# Set large zig-zag parameters
a_zigzag.zigzag_amplitude = 3000.0
a_zigzag.zigzag_frequency = 8.0
a_zigzag.trajectory = a_zigzag.generate_trajectory()

# 3. Zig-zag with noise
a_zigzag_noise = Attacker(
    start_position=start,
    waypoints=[mid],
    target_position=target,
    use_dynamic_trajectory=True,
    trajectory_aggressiveness=2.0,
    steps=600,
    noise_std=60.0  # 60m noise
)
# Set same large zig-zag parameters
a_zigzag_noise.zigzag_amplitude = 3000.0
a_zigzag_noise.zigzag_frequency = 8.0
a_zigzag_noise.trajectory = a_zigzag_noise.generate_trajectory()

# Plot all three
traj_linear = np.array(a_linear.trajectory)
traj_zigzag = np.array(a_zigzag.trajectory)
traj_zigzag_noise = np.array(a_zigzag_noise.trajectory)

plt.figure(figsize=(14, 8))
plt.plot(traj_linear[:, 0], traj_linear[:, 1], '-', color='gray', linewidth=1.5, alpha=0.7, label='Linear (no zig-zag, no noise)')
plt.plot(traj_zigzag[:, 0], traj_zigzag[:, 1], '-', color='blue', linewidth=1.5, alpha=0.8, label='Zig-zag (amplitude=3000m, freq=8, no noise)')
plt.plot(traj_zigzag_noise[:, 0], traj_zigzag_noise[:, 1], '-', color='red', linewidth=1.5, alpha=0.8, label='Zig-zag + noise (60m std)')

plt.scatter([start[0]], [start[1]], c='green', s=100, marker='o', label='Start', zorder=10)
plt.scatter([mid[0]], [mid[1]], c='orange', s=100, marker='s', label='Waypoint', zorder=10)
plt.scatter([target[0]], [target[1]], c='red', s=100, marker='X', label='Target', zorder=10)

plt.xlabel('X (meters)', fontsize=12)
plt.ylabel('Y (meters)', fontsize=12)
plt.title('Trajectory Comparison: Linear vs Zig-Zag vs Zig-Zag+Noise', fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(alpha=0.3)
plt.axis('equal')
plt.tight_layout()

output_file = 'quick_test_zigzag_noise.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved comparison plot to: {output_file}")
print(f"\nTrajectory stats:")
print(f"  Linear:          {len(traj_linear)} points")
print(f"  Zig-zag:         {len(traj_zigzag)} points")
print(f"  Zig-zag + noise: {len(traj_zigzag_noise)} points")

# Compute perpendicular offsets to quantify zig-zag amplitude
def compute_perp_offsets(traj, start, target):
    start_np = np.array(start, dtype=float)
    end_np = np.array(target, dtype=float)
    main_dir = end_np - start_np
    main_len = np.linalg.norm(main_dir)
    main_unit = main_dir / (main_len + 1e-12)
    perp = np.array([-main_unit[1], main_unit[0]])
    offsets = [np.dot(np.array(p) - start_np, perp) for p in traj]
    return offsets

offsets_linear = compute_perp_offsets(traj_linear, start, target)
offsets_zigzag = compute_perp_offsets(traj_zigzag, start, target)
offsets_zigzag_noise = compute_perp_offsets(traj_zigzag_noise, start, target)

print(f"\nPerpendicular offset range (meters):")
print(f"  Linear:          [{min(offsets_linear):.1f}, {max(offsets_linear):.1f}]")
print(f"  Zig-zag:         [{min(offsets_zigzag):.1f}, {max(offsets_zigzag):.1f}]")
print(f"  Zig-zag + noise: [{min(offsets_zigzag_noise):.1f}, {max(offsets_zigzag_noise):.1f}]")

print(f"\n🎯 If zig-zag is working correctly, you should see:")
print(f"   - Gray line: Nearly straight (small offsets)")
print(f"   - Blue line: Clear sinusoidal oscillation (large offsets)")
print(f"   - Red line: Oscillation with additional random jitter from noise")

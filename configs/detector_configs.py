from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DetectorType(Enum):
    RADAR = "radar"
    ACOUSTIC = "acoustic"
    VISUAL = "visual"
    RADIO = "radio"


@dataclass(frozen=True)
class BaseConfig:
    """Configuration for an acoustic detector."""

    radius: float = 5

    def probability(self, distance: float) -> float:
        """
        Simple linear falloff:
        - p(0)   = 1.0
        - p(r)   = 0.0
        - p(d)   = 0.0 for d >= r
        """
        # Vectorized: accept scalar or numpy array
        arr = np.asarray(distance)
        clipped = np.maximum(arr, 0.0)
        p = np.zeros_like(clipped, dtype=float)
        mask = clipped < self.radius
        p[mask] = 1.0 - (clipped[mask] / self.radius)
        p = np.clip(p, 0.0, 1.0)
        # Return scalar if input was scalar
        if np.ndim(distance) == 0:
            return float(p)
        return p

@dataclass(frozen=True)
class AcousticDetectorConfig:
    """Configuration for an acoustic detector."""

    radius: float = 20

    def probability(self, distance: float) -> float:
        """
        Simple linear falloff:
        - p(0)   = 1.0
        - p(r)   = 0.0
        - p(d)   = 0.0 for d >= r
        """
        arr = np.asarray(distance)
        clipped = np.maximum(arr, 0.0)
        p = np.zeros_like(clipped, dtype=float)
        mask = clipped < self.radius
        p[mask] = 1.0 - (clipped[mask] / self.radius)
        p = np.clip(p, 0.0, 1.0)
        if np.ndim(distance) == 0:
            return float(p)
        return p


@dataclass(frozen=True)
class VisualDetectorConfig:
    """Configuration for a visual detector."""

    radius: float = 1500
    distance_probability_map: dict = field(default_factory=lambda: {0: 1.0, 600: 0.85, 1000: 0.3, 1500: 0.0})

    def probability(self, distance: float) -> float:
        """
        Piecewise probability function based on configurable distance/probability points.
        Default values:
        - p(0)    = 1.0 (100%)
        - p(500)  = 0.8 (80%)
        - p(1000) = 0.3 (30%)
        - p(1500) = 0.0 (0%)
        - p(d)    = 0.0 for d >= radius
        
        Uses quadratic interpolation (slow at first) from 0 to 500m,
        linear interpolation from 500m to 1000m,
        and exponential decay from 1000m to 1500m.
        """
        arr = np.asarray(distance)
        clipped = np.maximum(arr, 0.0)
        p = np.zeros_like(clipped, dtype=float)
        
        # Get sorted distance points
        distances = sorted(self.distance_probability_map.keys())
        probabilities = [self.distance_probability_map[d] for d in distances]
        
        # Handle distances beyond the maximum defined distance
        max_distance = max(distances)
        mask_within_range = clipped <= max_distance
        
        # For each distance value, interpolate between the defined points
        for i in range(len(distances) - 1):
            d1, p1 = distances[i], probabilities[i]
            d2, p2 = distances[i + 1], probabilities[i + 1]
            
            # Check if this is the last segment (should use exponential decay)
            is_last_segment = (i == len(distances) - 2)
            
            # Find distances in this segment
            if is_last_segment:
                mask_segment = (clipped >= d1) & (clipped <= d2) & mask_within_range
            else:
                mask_segment = (clipped >= d1) & (clipped < d2) & mask_within_range
            
            if np.any(mask_segment):
                # Use exponential decay for the segment from 1000m to 1500m (last segment)
                if is_last_segment and d1 >= 1000:
                    # Exponential decay from p1 at d1 to 0.0 at d2
                    # Use normalized exponential decay: p(d) = p1 * (exp(-k*(d-d1)/(d2-d1)) - exp(-k)) / (1 - exp(-k))
                    # This ensures p(d1) = p1 and p(d2) = 0
                    k = 5.0  # Decay rate (higher = faster decay)
                    d_offset = clipped[mask_segment] - d1
                    d_range = d2 - d1
                    if d_range > 0:
                        # Normalized exponential decay that reaches 0 at d2
                        exp_term = np.exp(-k * d_offset / d_range)
                        exp_end = np.exp(-k)
                        p[mask_segment] = p1 * (exp_term - exp_end) / (1.0 - exp_end)
                    else:
                        p[mask_segment] = p1
                # Use quadratic (slow at first) for the segment from 0 to 500m
                elif d1 == 0:
                    # Quadratic interpolation: p(d) = p1 + (p2 - p1) * (d/d_range)^2
                    # This decreases slowly at first, then faster
                    if d2 != d1:  # Avoid division by zero
                        d_offset = clipped[mask_segment] - d1
                        d_range = d2 - d1
                        # Normalized distance squared: (d/d_range)^2
                        normalized_dist_sq = (d_offset / d_range) ** 2
                        p[mask_segment] = p1 + (p2 - p1) * normalized_dist_sq
                    else:
                        p[mask_segment] = p1
                else:
                    # Linear interpolation for other segments: p(d) = p1 + (p2 - p1) * (d - d1) / (d2 - d1)
                    if d2 != d1:  # Avoid division by zero
                        p[mask_segment] = p1 + (p2 - p1) * (clipped[mask_segment] - d1) / (d2 - d1)
                    else:
                        p[mask_segment] = p1
        
        # Handle distances beyond radius
        mask_beyond = clipped > self.radius
        if np.any(mask_beyond):
            p[mask_beyond] = 0.0
        
        p = np.clip(p, 0.0, 1.0)
        if np.ndim(distance) == 0:
            return float(p)
        return p


@dataclass(frozen=True)
class RadarDetectorConfig:
    """Configuration for a radar detector."""

    radius: float = 15

    def probability(self, distance: float) -> float:
        """
        Smooth exponential-like falloff, clamped so that:
        - p(0)   ~ 1.0
        - p(r)   = 0.0
        - p(d)   = 0.0 for d >= r
        """
        arr = np.asarray(distance)
        clipped = np.maximum(arr, 0.0)
        p = np.zeros_like(clipped, dtype=float)
        mask = clipped < self.radius
        if np.any(mask):
            k = 4.0
            x = np.zeros_like(clipped, dtype=float)
            x[mask] = clipped[mask] / self.radius
            p[mask] = np.exp(-k * x[mask])
        p = np.clip(p, 0.0, 1.0)
        if np.ndim(distance) == 0:
            return float(p)
        return p


# if __name__ == "__main__":
#     # Create VisualDetectorConfig instance
#     visual_config = VisualDetectorConfig()
    
#     # Create figure with subplots
#     fig = plt.figure(figsize=(15, 5))
    
#     # 1D plot: Probability vs Distance
#     ax1 = plt.subplot(1, 3, 1)
#     distances_1d = np.linspace(0, 2200, 1000)
#     probabilities_1d = visual_config.probability(distances_1d)
    
#     ax1.plot(distances_1d, probabilities_1d, 'b-', linewidth=2, label='Probability')
#     ax1.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='p(0) = 100%')
#     ax1.axvline(x=500, color='purple', linestyle='--', alpha=0.5, label='p(500) = 70%')
#     ax1.axvline(x=1000, color='orange', linestyle='--', alpha=0.5, label='p(1000) = 30%')
#     ax1.axvline(x=1500, color='r', linestyle='--', alpha=0.5, label='p(1500) = 0%')
#     ax1.axhline(y=1.0, color='g', linestyle=':', alpha=0.3)
#     ax1.axhline(y=0.7, color='purple', linestyle=':', alpha=0.3)
#     ax1.axhline(y=0.3, color='orange', linestyle=':', alpha=0.3)
#     ax1.axhline(y=0.0, color='r', linestyle=':', alpha=0.3)
    
#     # Mark key points
#     ax1.plot(0, visual_config.probability(0), 'go', markersize=10, label='p(0) = 100%')
#     ax1.plot(500, visual_config.probability(500), 'o', color='purple', markersize=10, label='p(500) = 70%')
#     ax1.plot(1000, visual_config.probability(1000), 'o', color='orange', markersize=10, label='p(1000) = 30%')
#     ax1.plot(1500, visual_config.probability(1500), 'ro', markersize=10, label='p(1500) = 0%')
    
#     ax1.set_xlabel('Distance (meters)', fontsize=12)
#     ax1.set_ylabel('Probability', fontsize=12)
#     ax1.set_title('Visual Detector: Probability vs Distance', fontsize=14, fontweight='bold')
#     ax1.grid(True, alpha=0.3)
#     ax1.legend(loc='upper right', fontsize=9)
#     ax1.set_xlim(-50, 1700)
#     ax1.set_ylim(-0.05, 1.05)
    
#     # 2D heatmap: Probability distribution in 2D space
#     ax2 = plt.subplot(1, 3, 2)
#     size = 200
#     x = np.linspace(-1500, 1500, size)
#     y = np.linspace(-1500, 1500, size)
#     X, Y = np.meshgrid(x, y)
#     distances_2d = np.sqrt(X**2 + Y**2)
#     probabilities_2d = visual_config.probability(distances_2d)
    
#     im = ax2.imshow(probabilities_2d, extent=[-1500, 1500, -1500, 1500], 
#                     origin='lower', cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
#     ax2.set_xlabel('X Distance (meters)', fontsize=12)
#     ax2.set_ylabel('Y Distance (meters)', fontsize=12)
#     ax2.set_title('2D Probability Distribution\n(Heatmap)', fontsize=14, fontweight='bold')
    
#     # Add circular contours for key distances
#     circle_500 = plt.Circle((0, 0), 500, fill=False, color='purple', linewidth=2, linestyle='--', label='500m (70%)')
#     circle_1000 = plt.Circle((0, 0), 1000, fill=False, color='orange', linewidth=2, linestyle='--', label='1000m (30%)')
#     circle_1500 = plt.Circle((0, 0), 1500, fill=False, color='red', linewidth=2, linestyle='--', label='1500m (0%)')
#     ax2.add_patch(circle_500)
#     ax2.add_patch(circle_1000)
#     ax2.add_patch(circle_1500)
    
#     plt.colorbar(im, ax=ax2, label='Probability', fraction=0.046, pad=0.04)
#     ax2.legend(loc='upper right', fontsize=9)
#     ax2.grid(True, alpha=0.3)
    
#     # 3D surface plot
#     ax3 = plt.subplot(1, 3, 3, projection='3d')
#     # Use a smaller grid for 3D plot for better performance
#     size_3d = 100
#     x_3d = np.linspace(-1500, 1500, size_3d)
#     y_3d = np.linspace(-1500, 1500, size_3d)
#     X_3d, Y_3d = np.meshgrid(x_3d, y_3d)
#     distances_3d = np.sqrt(X_3d**2 + Y_3d**2)
#     probabilities_3d = visual_config.probability(distances_3d)
    
#     surf = ax3.plot_surface(X_3d, Y_3d, probabilities_3d, cmap='RdYlGn', 
#                            linewidth=0, antialiased=True, vmin=0, vmax=1, alpha=0.9)
#     ax3.set_xlabel('X Distance (meters)', fontsize=10)
#     ax3.set_ylabel('Y Distance (meters)', fontsize=10)
#     ax3.set_zlabel('Probability', fontsize=10)
#     ax3.set_title('3D Probability Surface', fontsize=14, fontweight='bold')
#     ax3.set_zlim(0, 1)
    
#     plt.colorbar(surf, ax=ax3, label='Probability', shrink=0.5, pad=0.1)
    
#     plt.tight_layout()
#     plt.savefig('visual_detector_probability.png', dpi=150, bbox_inches='tight')
#     print("Visualization saved as 'visual_detector_probability.png'")
#     print(f"\nVerification:")
#     print(f"  p(0) = {visual_config.probability(0):.4f} (expected: 1.0)")
#     print(f"  p(500) = {visual_config.probability(500):.4f} (expected: 0.7)")
#     print(f"  p(1000) = {visual_config.probability(1000):.4f} (expected: 0.3)")
#     print(f"  p(1500) = {visual_config.probability(1500):.4f} (expected: 0.0)")
#     plt.show()


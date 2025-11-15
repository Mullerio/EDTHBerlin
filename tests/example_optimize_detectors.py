#!/usr/bin/env python3
"""
Example script demonstrating how to use the detector position optimization functions.

This script shows realistic scenarios with:
1. Large areas (5-15 km scale)
2. Realistic detection ranges (1500m radius)
3. Complex irregular shapes
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import matplotlib.pyplot as plt
from src.detectors import optimize_detector_positions
from configs.detector_configs import VisualDetectorConfig

# Example 1: Large Border Security Area with Complex Terrain
def example_border_corridor():
    """Optimize detector placement for a winding border corridor."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Border Corridor (10km x 8km area)")
    print("="*60)
    
    # Create a large grid: 100x80 cells, each 100m
    # Total area: 10,000m x 8,000m = 10km x 8km
    area_grid = np.ones((80, 100))
    
    # Define complex winding corridor (border patrol zone)
    # Create a winding path with varying width
    for x in range(100):
        # Create sine wave pattern for realistic border
        y_center = 40 + int(15 * np.sin(x / 8))
        width = 8 + int(3 * np.sin(x / 5))  # Varying width
        
        y_start = max(0, y_center - width)
        y_end = min(80, y_center + width)
        area_grid[y_start:y_end, x] = 0
    
    # Add some gaps/obstacles (mountains, restricted zones)
    area_grid[35:45, 30:35] = 1
    area_grid[30:50, 65:70] = 1
    
    print(f"Area to cover: ~{np.sum(area_grid == 0) * 0.01:.1f} km²")
    
    # Visual detectors with 1500m radius need more detectors
    n_detectors = 10
    
    # Optimize positions
    positions, metrics = optimize_detector_positions(
        area_grid=area_grid,
        n_detectors=n_detectors,
        cell_size=100.0,  # Each cell is 100 meters
        method='greedy+refine'  # Use greedy for large area
    )
    
    print(f"\nOptimal positions:")
    for i, (x, y) in enumerate(positions[:5], 1):
        print(f"  Detector {i}: ({x:.0f}m, {y:.0f}m)")
    print(f"  ... and {len(positions)-5} more detectors")
    
    # Visualize
    visualize_coverage(area_grid, positions, metrics, 
                      cell_size=100.0, title="Border Corridor Coverage (10km x 8km)")
    
    return positions, metrics


# Example 2: Urban District with Multiple Zones
def example_urban_district():
    """Optimize detector placement for a complex urban area with multiple districts."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Urban District (15km x 15km area)")
    print("="*60)
    
    # Large urban grid: 150x150 cells, each 100m
    # Total area: 15km x 15km
    area_grid = np.ones((150, 150))
    
    # Create multiple irregular zones (districts to monitor)
    
    # District 1: Large irregular polygon (downtown)
    def create_polygon(center, radius, n_points, irregularity):
        """Create irregular polygon."""
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        angles += np.random.uniform(-irregularity, irregularity, n_points)
        radii = radius * (1 + np.random.uniform(-0.3, 0.3, n_points))
        
        points = []
        for angle, r in zip(angles, radii):
            x = int(center[0] + r * np.cos(angle))
            y = int(center[1] + r * np.sin(angle))
            points.append((x, y))
        return points
    
    def fill_polygon(grid, points):
        """Fill polygon using scan-line algorithm."""
        from matplotlib.path import Path
        height, width = grid.shape
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        
        path = Path(points)
        mask = path.contains_points(coords)
        mask = mask.reshape(height, width)
        grid[mask] = 0
        return grid
    
    # District 1: Downtown (center)
    poly1 = create_polygon((75, 75), 25, 8, 0.3)
    area_grid = fill_polygon(area_grid, poly1)
    
    # District 2: Industrial zone (northeast)
    poly2 = create_polygon((110, 110), 20, 6, 0.4)
    area_grid = fill_polygon(area_grid, poly2)
    
    # District 3: Residential area (southwest)
    poly3 = create_polygon((40, 40), 18, 7, 0.35)
    area_grid = fill_polygon(area_grid, poly3)
    
    # District 4: Narrow corridor connecting districts
    for i in range(60, 90):
        y = 60 + int(10 * np.sin(i / 8))
        area_grid[max(0, y-3):min(150, y+3), i] = 0
    
    # Add some obstacles/exclusion zones
    area_grid[70:80, 70:80] = 1  # Central exclusion (government building)
    area_grid[100:110, 100:110] = 1  # Airport
    
    print(f"Area to cover: ~{np.sum(area_grid == 0) * 0.01:.1f} km²")
    
    # Need many detectors for this large complex area
    n_detectors = 10
    
    # Optimize positions (use greedy for speed with large area)
    positions, metrics = optimize_detector_positions(
        area_grid=area_grid,
        n_detectors=n_detectors,
        cell_size=100.0,
        method='greedy'
    )
    
    print(f"\nOptimal positions (showing first 5):")
    for i, (x, y) in enumerate(positions[:5], 1):
        print(f"  Detector {i}: ({x:.0f}m, {y:.0f}m)")
    print(f"  ... and {len(positions)-5} more detectors")
    
    # Visualize
    visualize_coverage(area_grid, positions, metrics, 
                      cell_size=100.0, title="Urban District Coverage (15km x 15km)")
    
    return positions, metrics


# Example 3: Coastal Security Zone with Islands
def example_coastal_zone():
    """Optimize detector placement for coastal area with irregular coastline."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Coastal Security Zone (8km x 8km)")
    print("="*60)
    
    # Create grid: 80x80 cells, each 100m
    area_grid = np.ones((80, 80))
    
    # Create irregular coastline
    coastline_y = []
    for x in range(80):
        # Complex coastline with multiple frequencies
        y = 40 + int(12 * np.sin(x / 6) + 5 * np.cos(x / 3) + 3 * np.sin(x / 10))
        coastline_y.append(y)
    
    # Fill coastal monitoring zone (band along coast)
    for x in range(80):
        y_coast = coastline_y[x]
        # Monitoring zone extends inland
        y_start = max(0, y_coast - 15)
        y_end = min(80, y_coast + 8)
        area_grid[y_start:y_end, x] = 0
    
    # Add some small islands (separate areas to monitor)
    def add_island(cx, cy, radius):
        for i in range(80):
            for j in range(80):
                if (i - cy)**2 + (j - cx)**2 <= radius**2:
                    area_grid[i, j] = 0
    
    add_island(20, 15, 5)
    add_island(55, 25, 6)
    add_island(70, 18, 4)
    
    # Add some exclusion zones (ports, naval bases)
    area_grid[35:42, 40:48] = 1
    
    print(f"Area to cover: ~{np.sum(area_grid == 0) * 0.01:.1f} km²")
    
    n_detectors = 30
    
    # Optimize positions
    positions, metrics = optimize_detector_positions(
        area_grid=area_grid,
        n_detectors=n_detectors,
        cell_size=100.0,
        method='greedy'
    )
    
    print(f"\nOptimal positions (showing first 5):")
    for i, (x, y) in enumerate(positions[:5], 1):
        print(f"  Detector {i}: ({x:.0f}m, {y:.0f}m)")
    print(f"  ... and {len(positions)-5} more detectors")
    
    # Visualize
    visualize_coverage(area_grid, positions, metrics, 
                      cell_size=100.0, title="Coastal Zone Coverage (8km x 8km)")
    
    return positions, metrics


def visualize_coverage(area_grid, positions, metrics, cell_size=1.0, title="Coverage Map"):
    """Visualize the detector placement and coverage."""
    coverage_map = metrics.get('coverage_map')
    
    if coverage_map is None:
        print("No coverage map to visualize")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Target Area
    ax1 = axes[0]
    ax1.imshow(area_grid, cmap='RdYlGn_r', origin='lower', 
              extent=[0, area_grid.shape[1]*cell_size, 0, area_grid.shape[0]*cell_size])
    ax1.set_title('Target Area\n(Green = area to cover)', fontweight='bold')
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Coverage Map with Detector Positions
    ax2 = axes[1]
    im = ax2.imshow(coverage_map, cmap='RdYlGn', origin='lower', vmin=0, vmax=1,
                   extent=[0, area_grid.shape[1]*cell_size, 0, area_grid.shape[0]*cell_size])
    
    # Plot detector positions
    for i, (x, y) in enumerate(positions, 1):
        ax2.plot(x, y, 'b*', markersize=20, markeredgecolor='white', markeredgewidth=2)
        ax2.text(x, y, str(i), ha='center', va='center', color='white', 
                fontweight='bold', fontsize=10)
    
    ax2.set_title(f'Coverage Map with Detectors\n(Mean: {metrics["mean_coverage"]:.1%})', 
                 fontweight='bold')
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax2, label='Detection Probability')
    
    # Plot 3: Coverage in Target Area Only
    ax3 = axes[2]
    masked_coverage = coverage_map.copy()
    masked_coverage[area_grid != 0] = np.nan  # Hide non-target areas
    
    im3 = ax3.imshow(masked_coverage, cmap='RdYlGn', origin='lower', vmin=0, vmax=1,
                    extent=[0, area_grid.shape[1]*cell_size, 0, area_grid.shape[0]*cell_size])
    
    # Plot detector positions
    for i, (x, y) in enumerate(positions, 1):
        ax3.plot(x, y, 'b*', markersize=20, markeredgecolor='white', markeredgewidth=2)
    
    ax3.set_title(f'Coverage in Target Area Only\n(Min: {metrics["min_coverage"]:.1%})', 
                 fontweight='bold')
    ax3.set_xlabel('X (meters)')
    ax3.set_ylabel('Y (meters)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(im3, ax=ax3, label='Detection Probability')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("REALISTIC DETECTOR POSITION OPTIMIZATION EXAMPLES")
    print("="*60)
    print("\nThese examples demonstrate optimization for:")
    print("  - Large-scale areas (5-15 km)")
    print("  - Realistic detection ranges (1500m radius)")
    print("  - Complex irregular shapes")
    print("  - Multiple disconnected zones")
    print("="*60)
    
    # Run examples
    print("\nRunning Example 1...")
    example_border_corridor()
    
    print("\nRunning Example 2...")
    example_urban_district()
    
    print("\nRunning Example 3...")
    example_coastal_zone()
    
    print("\n" + "="*60)
    print("All examples complete!")
    print("="*60 + "\n")


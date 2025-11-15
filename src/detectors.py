import configs.detector_configs as detector_configs
import numpy as np
from scipy.optimize import minimize

class Detector:
    def __init__(self, type: detector_configs.DetectorType, position: tuple):
        self.type = type
        self.position = position        
        
        if type == detector_configs.DetectorType.RADAR:
            self.config = detector_configs.RadarDetectorConfig()
        elif type == detector_configs.DetectorType.ACOUSTIC:
            self.config = detector_configs.AcousticDetectorConfig()
        elif type == detector_configs.DetectorType.VISUAL:
            self.config = detector_configs.VisualDetectorConfig()
        elif type == detector_configs.DetectorType.RADIO:
            self.config = detector_configs.RadioDetectorConfig()
        else:
            self.config = detector_configs.BaseConfig()


        self.radius = self.config.radius
        self.probability_function = self.config.probability

    def probability(self, distance: float) -> float:
        return self.probability_function(distance)

    def detect(self, target) -> bool:
        return self.probability(target.distance) > 0.5
    
    
def Rect_Detectors(detector_type: detector_configs.DetectorType, grid_size: tuple, spacing: float):
    """
    Create a rectangular grid of detectors defined by corner points or dimensions.

    Parameters:
    - detector_type: DetectorType enum value
    - grid_size: Either:
                 * Four (x,y) corner tuples defining rectangle bounds (in physical meters)
                 * Legacy: (width, height) tuple (assumes corners at (0,0), (width,0), (width,height), (0,height))
    - spacing: spacing between adjacent detectors in physical units (meters)

    Returns:
        List of `Detector` instances with positions in physical coordinates (meters).
    
    All detector positions are in continuous physical coordinates and do NOT depend on
    environment grid resolution.
    """
    # New signature: second argument is corners (iterable of 4 (x,y) pairs)
    # Accept either a single tuple-of-4 or the old (width,height) for backward compatibility
    detectors = []
    # Backward compatibility: if grid_size is a pair of ints assume full-grid width,height
    try:
        # If grid_size looks like four corner points (len==4 and each has 2 items)
        if hasattr(grid_size, '__len__') and len(grid_size) == 4 and hasattr(grid_size[0], '__len__'):
            corners = grid_size
            xs = [float(c[0]) for c in corners]
            ys = [float(c[1]) for c in corners]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
        else:
            # Legacy behavior: grid_size was (width, height)
            width, height = grid_size
            x_min, x_max = 0.0, float(width)
            y_min, y_max = 0.0, float(height)
    except Exception:
        # Fallback to using grid_size as width,height
        width, height = grid_size
        x_min, x_max = 0.0, float(width)
        y_min, y_max = 0.0, float(height)

    # Create ranges with the requested spacing (include endpoint)
    if spacing <= 0:
        raise ValueError('spacing must be positive')
    xs = np.arange(x_min, x_max + 1e-8, spacing)
    ys = np.arange(y_min, y_max + 1e-8, spacing)

    for x in xs:
        for y in ys:
            detector = Detector(type=detector_type, position=(float(x), float(y)))
            detectors.append(detector)

    return detectors


def Triang_Detectors(detector_type: detector_configs.DetectorType, triangle_corners: tuple, spacing: float):
    """
    Generate detectors uniformly inside a triangle defined by three corner points.

    Parameters:
    - detector_type: DetectorType enum value
    - triangle_corners: iterable of three (x,y) tuples in physical coordinates (meters)
    - spacing: approximate spacing between detectors in physical units (meters)

    Returns:
        List of `Detector` instances with positions in physical coordinates (meters).
    
    All detector positions are in continuous physical coordinates and do NOT depend on
    environment grid resolution.
    """

    if spacing <= 0:
        raise ValueError('spacing must be positive')

    if not (hasattr(triangle_corners, '__len__') and len(triangle_corners) == 3):
        raise ValueError('triangle_corners must be an iterable of three (x,y) pairs')

    (x1, y1), (x2, y2), (x3, y3) = [(float(a), float(b)) for a, b in triangle_corners]

    x_min = min(x1, x2, x3)
    x_max = max(x1, x2, x3)
    y_min = min(y1, y2, y3)
    y_max = max(y1, y2, y3)

    xs = np.arange(x_min, x_max + 1e-8, spacing)
    ys = np.arange(y_min, y_max + 1e-8, spacing)

    # Barycentric method to test if point is inside triangle
    def _point_in_triangle(px, py):
        # Compute barycentric coordinates
        denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
        if denom == 0:
            return False
        a = ((y2 - y3)*(px - x3) + (x3 - x2)*(py - y3)) / denom
        b = ((y3 - y1)*(px - x3) + (x1 - x3)*(py - y3)) / denom
        c = 1.0 - a - b
        return (a >= 0) and (b >= 0) and (c >= 0)

    detectors = []
    for x in xs:
        for y in ys:
            if _point_in_triangle(float(x), float(y)):
                detectors.append(Detector(type=detector_type, position=(float(x), float(y))))

    return detectors
            

def set_nonobservable_triangle(sector_env, triangle_corners: tuple, physical_coords=True):
    """
    Mark the grid cells inside a triangle as non-observable on the given SectorEnv.

    Args:
        sector_env: instance of SectorEnv (or Environment with observable_mask)
        triangle_corners: iterable of three (x,y) tuples defining the triangle
        physical_coords: if True (default), triangle_corners are in physical meters;
                        if False, they are in grid cell indices
    
    This helper modifies `sector_env.observable_mask` in-place, setting cells
    whose center point lies inside the triangle to False (non-observable).
    """
    if not (hasattr(triangle_corners, '__len__') and len(triangle_corners) == 3):
        raise ValueError('triangle_corners must be an iterable of three (x,y) pairs')

    # Convert to grid coordinates if needed
    if physical_coords:
        cell_size = getattr(sector_env, 'cell_size', 1.0)
        (x1, y1), (x2, y2), (x3, y3) = [(float(a) / cell_size, float(b) / cell_size) for a, b in triangle_corners]
    else:
        (x1, y1), (x2, y2), (x3, y3) = [(float(a), float(b)) for a, b in triangle_corners]

    # Bounding box to limit work (in grid cell indices)
    x_min = max(0, int(np.floor(min(x1, x2, x3))))
    x_max = min(int(np.ceil(max(x1, x2, x3))), sector_env.width - 1)
    y_min = max(0, int(np.floor(min(y1, y2, y3))))
    y_max = min(int(np.ceil(max(y1, y2, y3))), sector_env.height - 1)

    # barycentric helper
    def _point_in_triangle(px, py):
        denom = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
        if denom == 0:
            return False
        a = ((y2 - y3)*(px - x3) + (x3 - x2)*(py - y3)) / denom
        b = ((y3 - y1)*(px - x3) + (x1 - x3)*(py - y3)) / denom
        c = 1.0 - a - b
        return (a >= 0) and (b >= 0) and (c >= 0)

    # Iterate integer grid cells and mark those whose center point is inside
    for yi in range(y_min, y_max + 1):
        for xi in range(x_min, x_max + 1):
            cx = float(xi) + 0.5
            cy = float(yi) + 0.5
            if _point_in_triangle(cx, cy):
                try:
                    sector_env.observable_mask[yi, xi] = False
                except Exception:
                    # best-effort: if mask not present or indexing fails, skip
                    continue


def compute_coverage_map(detector_positions, detector_config, grid_shape, cell_size=1.0):
    """
    Compute the combined probability coverage map for given detector positions.
    
    Parameters:
    - detector_positions: List of (x, y) tuples in physical coordinates (meters)
    - detector_config: A detector configuration object with a probability() method
    - grid_shape: (height, width) of the grid
    - cell_size: Size of each grid cell in physical units (meters)
    
    Returns:
    - coverage_map: 2D array of shape grid_shape with combined probabilities
    
    The combined probability uses the probabilistic OR formula:
    P(detected by at least one) = 1 - prod(1 - P(detected by i))
    """
    height, width = grid_shape
    coverage_map = np.zeros((height, width), dtype=float)
    
    if len(detector_positions) == 0:
        return coverage_map
    
    # Create grid of cell centers in physical coordinates
    y_coords, x_coords = np.meshgrid(
        np.arange(height) * cell_size + cell_size / 2,
        np.arange(width) * cell_size + cell_size / 2,
        indexing='ij'
    )
    
    # Compute combined probability using probabilistic OR
    # P(at least one detects) = 1 - P(none detect) = 1 - prod(1 - p_i)
    prob_none_detect = np.ones((height, width), dtype=float)
    
    for det_x, det_y in detector_positions:
        # Compute distances from this detector to all grid cells
        distances = np.sqrt((x_coords - det_x)**2 + (y_coords - det_y)**2)
        
        # Get detection probabilities
        probs = detector_config.probability(distances)
        
        # Update combined probability
        prob_none_detect *= (1.0 - probs)
    
    coverage_map = 1.0 - prob_none_detect
    
    return coverage_map


def evaluate_coverage(detector_positions, area_grid, detector_config, cell_size=1.0, 
                     coverage_threshold=0.5):
    """
    Evaluate the quality of detector placement.
    
    Parameters:
    - detector_positions: List of (x, y) tuples in physical coordinates
    - area_grid: 2D array where 0 indicates area to cover, 1 indicates outside area
    - detector_config: A detector configuration object
    - cell_size: Size of each grid cell in physical units (meters)
    - coverage_threshold: Minimum probability to consider a cell "covered"
    
    Returns:
    - score: Dictionary with various coverage metrics
    """
    grid_shape = area_grid.shape
    coverage_map = compute_coverage_map(detector_positions, detector_config, 
                                       grid_shape, cell_size)
    
    # Mask for area to cover (where area_grid == 0)
    target_mask = (area_grid == 0)
    
    if not np.any(target_mask):
        return {
            'mean_coverage': 0.0,
            'min_coverage': 0.0,
            'fraction_covered': 0.0,
            'total_coverage': 0.0
        }
    
    # Extract coverage values only for target area
    target_coverage = coverage_map[target_mask]
    
    # Compute metrics
    mean_coverage = np.mean(target_coverage)
    min_coverage = np.min(target_coverage)
    fraction_covered = np.mean(target_coverage >= coverage_threshold)
    total_coverage = np.sum(target_coverage)
    
    return {
        'mean_coverage': mean_coverage,
        'min_coverage': min_coverage,
        'fraction_covered': fraction_covered,
        'total_coverage': total_coverage,
        'coverage_map': coverage_map
    }


def optimize_detector_positions_greedy(area_grid, n_detectors, detector_config, 
                                      cell_size=1.0, n_candidates=100):
    """
    Use a greedy algorithm to find good detector positions.
    
    Places detectors one at a time, each time choosing the position that
    maximally improves coverage of the target area.
    
    Parameters:
    - area_grid: 2D array where 0 indicates area to cover, 1 indicates outside
    - n_detectors: Number of detectors to place
    - detector_config: A detector configuration object
    - cell_size: Size of each grid cell in physical units (meters)
    - n_candidates: Number of candidate positions to evaluate at each step
    
    Returns:
    - positions: List of (x, y) tuples in physical coordinates
    """
    height, width = area_grid.shape
    target_mask = (area_grid == 0)
    
    if not np.any(target_mask):
        print("Warning: No target area found (no zeros in area_grid)")
        return []
    
    # Get coordinates of target cells
    target_coords = np.argwhere(target_mask)  # Returns (row, col) pairs
    
    # Convert to physical coordinates (center of cells)
    target_physical = np.column_stack([
        target_coords[:, 1] * cell_size + cell_size / 2,  # x (col)
        target_coords[:, 0] * cell_size + cell_size / 2   # y (row)
    ])
    
    positions = []
    
    # Create grid of cell centers for coverage computation
    y_coords, x_coords = np.meshgrid(
        np.arange(height) * cell_size + cell_size / 2,
        np.arange(width) * cell_size + cell_size / 2,
        indexing='ij'
    )
    
    # Track cumulative "probability none detect"
    prob_none_detect = np.ones((height, width), dtype=float)
    
    for i in range(n_detectors):
        best_position = None
        best_score = -np.inf
        
        # Generate candidate positions (sample from target area + some random)
        if len(target_physical) > n_candidates:
            candidate_indices = np.random.choice(len(target_physical), 
                                                size=n_candidates, replace=False)
            candidates = target_physical[candidate_indices]
        else:
            candidates = target_physical.copy()
        
        # Evaluate each candidate
        for cand_x, cand_y in candidates:
            # Compute distances from this candidate to all grid cells
            distances = np.sqrt((x_coords - cand_x)**2 + (y_coords - cand_y)**2)
            
            # Get detection probabilities
            probs = detector_config.probability(distances)
            
            # Compute new combined probability if we add this detector
            new_prob_none = prob_none_detect * (1.0 - probs)
            new_coverage = 1.0 - new_prob_none
            
            # Score: focus on improving coverage in target area
            target_coverage = new_coverage[target_mask]
            score = np.sum(target_coverage)  # Total coverage in target area
            
            if score > best_score:
                best_score = score
                best_position = (cand_x, cand_y)
        
        if best_position is not None:
            positions.append(best_position)
            
            # Update cumulative coverage
            det_x, det_y = best_position
            distances = np.sqrt((x_coords - det_x)**2 + (y_coords - det_y)**2)
            probs = detector_config.probability(distances)
            prob_none_detect *= (1.0 - probs)
            
            current_coverage = 1.0 - prob_none_detect
            mean_cov = np.mean(current_coverage[target_mask])
            print(f"Placed detector {i+1}/{n_detectors} at ({det_x:.1f}, {det_y:.1f}), "
                  f"mean coverage: {mean_cov:.3f}")
    
    return positions


def optimize_detector_positions_refined(area_grid, n_detectors, detector_config,
                                       cell_size=1.0, initial_positions=None,
                                       max_iterations=50):
    """
    Refine detector positions using local optimization.
    
    Uses gradient-free optimization (Nelder-Mead) to improve detector placement.
    
    Parameters:
    - area_grid: 2D array where 0 indicates area to cover
    - n_detectors: Number of detectors
    - detector_config: A detector configuration object
    - cell_size: Size of each grid cell in physical units (meters)
    - initial_positions: Starting positions (if None, uses greedy initialization)
    - max_iterations: Maximum optimization iterations
    
    Returns:
    - positions: List of (x, y) tuples in physical coordinates
    """
    height, width = area_grid.shape
    target_mask = (area_grid == 0)
    
    # Create grid of cell centers
    y_coords, x_coords = np.meshgrid(
        np.arange(height) * cell_size + cell_size / 2,
        np.arange(width) * cell_size + cell_size / 2,
        indexing='ij'
    )
    
    # Get initial positions
    if initial_positions is None:
        print("Running greedy initialization...")
        initial_positions = optimize_detector_positions_greedy(
            area_grid, n_detectors, detector_config, cell_size
        )
    
    if len(initial_positions) == 0:
        return []
    
    # Flatten positions for optimization
    x0 = np.array(initial_positions).flatten()
    
    # Define objective function (negative because we minimize)
    def objective(positions_flat):
        positions = positions_flat.reshape(-1, 2)
        
        # Compute coverage
        prob_none_detect = np.ones((height, width), dtype=float)
        
        for det_x, det_y in positions:
            distances = np.sqrt((x_coords - det_x)**2 + (y_coords - det_y)**2)
            probs = detector_config.probability(distances)
            prob_none_detect *= (1.0 - probs)
        
        coverage_map = 1.0 - prob_none_detect
        
        # Score: maximize mean coverage in target area, penalize low min coverage
        target_coverage = coverage_map[target_mask]
        mean_cov = np.mean(target_coverage)
        min_cov = np.min(target_coverage)
        
        # Objective: maximize mean + weight for min coverage
        score = mean_cov + 0.2 * min_cov
        
        return -score  # Negative because we minimize
    
    # Define bounds (keep detectors within grid)
    bounds = []
    for _ in range(n_detectors):
        bounds.append((0, width * cell_size))  # x bounds
        bounds.append((0, height * cell_size))  # y bounds
    
    print(f"\nRefining positions with local optimization...")
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iterations, 'disp': False}
    )
    
    # Extract optimized positions
    optimized_positions = result.x.reshape(-1, 2)
    positions_list = [(float(x), float(y)) for x, y in optimized_positions]
    
    return positions_list


def optimize_detector_positions(area_grid, n_detectors, detector_config=None,
                                cell_size=1.0, method='greedy+refine',
                                detector_type=None):
    """
    Find optimal positions for detectors to cover a target area.
    
    Parameters:
    - area_grid: 2D numpy array where 0 indicates area to cover, 1 indicates outside
    - n_detectors: Number of detectors to place
    - detector_config: Detector configuration object (if None, uses VisualDetectorConfig)
    - cell_size: Physical size of each grid cell in meters (default: 1.0)
    - method: Optimization method:
        * 'greedy': Fast greedy algorithm
        * 'refined': Greedy + local optimization (slower but better)
        * 'greedy+refine': Alias for 'refined' (default)
    - detector_type: DetectorType enum (used if detector_config is None)
    
    Returns:
    - positions: List of (x, y) tuples in physical coordinates (meters)
    - metrics: Dictionary with coverage metrics
    
    Example:
    >>> area = np.ones((100, 100))
    >>> area[20:80, 20:80] = 0  # Define area to cover
    >>> positions, metrics = optimize_detector_positions(area, n_detectors=5, cell_size=10.0)
    >>> print(f"Mean coverage: {metrics['mean_coverage']:.2%}")
    """
    # Get detector config
    if detector_config is None:
        if detector_type is None:
            detector_type = detector_configs.DetectorType.VISUAL
        
        if detector_type == detector_configs.DetectorType.VISUAL:
            detector_config = detector_configs.VisualDetectorConfig()
        elif detector_type == detector_configs.DetectorType.RADAR:
            detector_config = detector_configs.RadarDetectorConfig()
        elif detector_type == detector_configs.DetectorType.ACOUSTIC:
            detector_config = detector_configs.AcousticDetectorConfig()
        else:
            detector_config = detector_configs.BaseConfig()
    
    # Validate inputs
    if not isinstance(area_grid, np.ndarray) or area_grid.ndim != 2:
        raise ValueError("area_grid must be a 2D numpy array")
    
    if n_detectors <= 0:
        return [], {'mean_coverage': 0.0, 'min_coverage': 0.0, 
                   'fraction_covered': 0.0, 'total_coverage': 0.0}
    
    # Choose optimization method
    if method in ['greedy']:
        positions = optimize_detector_positions_greedy(
            area_grid, n_detectors, detector_config, cell_size
        )
    elif method in ['refined', 'greedy+refine']:
        positions = optimize_detector_positions_refined(
            area_grid, n_detectors, detector_config, cell_size
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'greedy' or 'refined'")
    
    # Evaluate final coverage
    metrics = evaluate_coverage(positions, area_grid, detector_config, cell_size)
    
    print(f"\n{'='*60}")
    print(f"Optimization complete:")
    print(f"  Mean coverage: {metrics['mean_coverage']:.2%}")
    print(f"  Min coverage: {metrics['min_coverage']:.2%}")
    print(f"  Fraction above 50%: {metrics['fraction_covered']:.2%}")
    print(f"{'='*60}\n")
    
    return positions, metrics

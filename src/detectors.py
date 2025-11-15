import configs.detector_configs as detector_configs
import numpy as np

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
            


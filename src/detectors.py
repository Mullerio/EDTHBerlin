import configs.detector_configs as detector_configs
import numpy as np

class Detector:
    def __init__(self, type: detector_configs.DetectorType, position: tuple):
        self.type = type
        self.position = position        
        
        if type == detector_configs.DetectorType.RADAR:
            self.config = detector_configs.RadarDetectorConfig()
        if type == detector_configs.DetectorType.ACOUSTIC:
            self.config = detector_configs.AcousticDetectorConfig()
        if type == detector_configs.DetectorType.VISUAL:
            self.config = detector_configs.VisualDetectorConfig()
        if type == detector_configs.DetectorType.RADIO:
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
    Create a grid of detectors of given type over the specified area.

    detector_type: Type of detectors to create.
    grid_size: Tuple (width, height) specifying the area size.
    spacing: Distance between adjacent detectors in the grid.
    """
    """
    Create a rectangular subgrid of detectors defined by the four corner points.

    Parameters
    - detector_type: DetectorType enum value
    - corners: iterable of four (x, y) corner tuples (order doesn't matter)
               The function will compute the bounding box of these corners and
               populate detectors inside that box on a regular spacing grid.
    - spacing: spacing between adjacent detectors in the subgrid (float)

    Returns a list of `Detector` instances placed on the subgrid (including edges).
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

    Parameters
    - detector_type: DetectorType enum value
    - triangle_corners: iterable of three (x,y) tuples
    - spacing: approximate spacing between detectors

    Returns a list of `Detector` instances placed approximately on a grid inside the triangle.
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
            


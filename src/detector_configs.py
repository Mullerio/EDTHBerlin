from dataclasses import dataclass
from math import exp

from src.detectors import DetectorType


@dataclass(frozen=True)
class AcousticDetectorConfig:
    """Configuration for an acoustic detector."""

    radius: float

    def probability(self, distance: float) -> float:
        """
        Simple linear falloff:
        - p(0)   = 1.0
        - p(r)   = 0.0
        - p(d)   = 0.0 for d >= r
        """
        d = max(0.0, float(distance))
        if d >= self.radius:
            return 0.0
        p = 1.0 - d / self.radius
        return max(0.0, min(1.0, p))


@dataclass(frozen=True)
class VisualDetectorConfig:
    """Configuration for a visual detector."""

    radius: float

    def probability(self, distance: float) -> float:
        """
        Quadratic falloff:
        - p(0)   = 1.0
        - p(r)   = 0.0
        - p(d)   = 0.0 for d >= r
        """
        d = max(0.0, float(distance))
        if d >= self.radius:
            return 0.0
        x = d / self.radius
        p = 1.0 - x * x
        return max(0.0, min(1.0, p))


@dataclass(frozen=True)
class RadarDetectorConfig:
    """Configuration for a radar detector."""

    radius: float

    def probability(self, distance: float) -> float:
        """
        Smooth exponential-like falloff, clamped so that:
        - p(0)   ~ 1.0
        - p(r)   = 0.0
        - p(d)   = 0.0 for d >= r
        """
        d = max(0.0, float(distance))
        if d >= self.radius:
            return 0.0

        # Shape parameter can be tuned; larger => steeper falloff
        k = 4.0
        x = d / self.radius
        p = exp(-k * x)  # in (0, 1] for x in [0, 1)
        return max(0.0, min(1.0, p))


# Example default configs that you can reuse in your simulation.

DEFAULT_ACOUSTIC = AcousticDetectorConfig(radius=500.0)
DEFAULT_VISUAL = VisualDetectorConfig(radius=300.0)
DEFAULT_RADAR = RadarDetectorConfig(radius=1000.0)


DEFAULT_DETECTOR_CONFIGS = {
    DetectorType.ACOUSTIC: DEFAULT_ACOUSTIC,
    DetectorType.VISUAL: DEFAULT_VISUAL,
    DetectorType.RADAR: DEFAULT_RADAR,
}



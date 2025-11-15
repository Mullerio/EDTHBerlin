from dataclasses import dataclass
from enum import Enum
import numpy as np


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

    radius: float = 10

    def probability(self, distance: float) -> float:
        """
        Quadratic falloff:
        - p(0)   = 1.0
        - p(r)   = 0.0
        - p(d)   = 0.0 for d >= r
        """
        arr = np.asarray(distance)
        clipped = np.maximum(arr, 0.0)
        p = np.zeros_like(clipped, dtype=float)
        mask = clipped < self.radius
        x = np.zeros_like(clipped, dtype=float)
        x[mask] = clipped[mask] / self.radius
        p[mask] = 1.0 - x[mask] * x[mask]
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
    





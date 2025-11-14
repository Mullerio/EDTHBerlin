from dataclasses import dataclass
from enum import Enum
from math import exp


class InterceptorType(Enum):
    A = "A"
    B = "B"
    
@dataclass(frozen=True)
class BaseInterceptorConfig:
    """Configuration for Interceptor A."""
    capacity: float = 10
    velocity: float = 100
    time_to_start: float = 20


@dataclass(frozen=True)
class InterceptorConfigA:
    """Configuration for Interceptor A."""
    capacity: float = 10
    velocity: float = 100
    time_to_start: float = 20



@dataclass(frozen=True)
class InterceptorConfigB:
    """Configuration for Interceptor B."""
    capacity: int = 10
    velocity: float = 50
    time_to_start: float = 10


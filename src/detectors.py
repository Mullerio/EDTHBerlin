from abc import ABC, abstractmethod
from enum import Enum

class DetectorType(Enum):
    RADAR = "radar"
    ACUSTIC = "acustic"
    VISUAL = "visual"
    RADIO = "radio"


class Detector:
    def __init__(self, type: DetectorType)
        self.type = type
        if type == DetectorType.RADAR:
            


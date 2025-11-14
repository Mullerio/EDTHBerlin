import src.detector_configs


class Detector:
    def __init__(self, type: DetectorType, position: tuple):
        self.type = type
        self.position = position
        
        if type == DetectorType.RADAR:
            self.config = RadarDetectorConfig()
        if type == DetectorType.ACOUSTIC:
            self.config = AcousticDetectorConfig()
        if type == DetectorType.VISUAL:
            self.config = VisualDetectorConfig()
        if type == DetectorType.RADIO:
            self.config = RadioDetectorConfig()

        self.radius = self.config.radius
        self.probability_function = self.config.probability

    def probability(self, distance: float) -> float:
        return self.probability_function(distance)

    def detect(self, target: Target) -> bool:
        return self.probability(target.distance) > 0.5

    

            


import configs.detector_configs as detector_configs


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

    

            


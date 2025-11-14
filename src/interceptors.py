import configs.interceptor_configs as interceptor_configs

class InterceptorUnit:
    def __init__(self, type: interceptor_configs.InterceptorType, position: tuple, capacity = None):
        self.type = type
        self.position = position
    
        if type == interceptor_configs.InterceptorType.A:
            self.config = interceptor_configs.InterceptorConfigA()
        if type == interceptor_configs.InterceptorType.B:
            self.config = interceptor_configs.InterceptorConfigB()
        else:
            self.config = interceptor_configs.BaseInterceptorConfig()

        self.velocity = self.config.velocity
        self.time_to_start = self.config.time_to_start

        if capacity is None:
            self.capacity = self.config.capacity
        else:
            self.capacity = capacity


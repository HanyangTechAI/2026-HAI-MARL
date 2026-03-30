import numpy as np
from .base_arm import BaseArm

class TrendArm(BaseArm):
    def __init__(self, arm_name, start_mean, slope, variance): 
        super().__init__(arm_name=arm_name)
        self.start_mean = start_mean
        self.slope = slope
        self.variance = variance

    def draw(self, t=None):
        if t is None: t = 0
        current_expected_value = max(0.0, self.start_mean + (self.slope * t))
        current_reward = np.random.normal(current_expected_value, np.sqrt(abs(self.variance)))
        return current_reward
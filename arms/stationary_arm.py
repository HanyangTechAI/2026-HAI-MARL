import numpy as np
from .base_arm import BaseArm

class StationaryArm(BaseArm):
    def __init__(self, arm_name, mean, variance):
        super().__init__(arm_name=arm_name)
        self.mean = mean
        self.variance = variance

    def draw(self, t=None):
        current_reward = np.random.normal(self.mean, np.sqrt(abs(self.variance)))
        return max(0.0, current_reward)
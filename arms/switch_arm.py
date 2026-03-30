import numpy as np
import pandas as pd
from .base_arm import BaseArm

class SwitchArm(BaseArm):
    def __init__(self, arm_name, base_mean, base_variance, switch_csv):
        super().__init__(arm_name=arm_name)
        self.base_mean = base_mean
        self.base_variance = base_variance
        
        self.switch_schedule = {}
        try:
            df_switches = pd.read_csv(switch_csv)
            my_switches = df_switches[df_switches['arm_name'] == arm_name].sort_values(by='switch_step')
            self.switch_schedule = dict(zip(my_switches['switch_step'], my_switches['multiplier']))
        except FileNotFoundError:
            pass

    def draw(self, t=None):
        if t is None: t = 1
        current_multiplier = 1.0
        for step, multiplier in self.switch_schedule.items():
            if t >= step: current_multiplier = multiplier
            else: break
                
        current_expected_mean = self.base_mean * current_multiplier
        current_variance = self.base_variance * (current_multiplier ** 2)
        current_reward = np.random.normal(current_expected_mean, np.sqrt(abs(current_variance)))
        
        return max(0.0, current_reward)
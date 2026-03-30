import numpy as np
import pandas as pd
from .base_arm import BaseArm

class EventShockArm(BaseArm):
    def __init__(self, arm_name, base_mean, base_variance, shocks_csv, season_csv):
        super().__init__(arm_name=arm_name)
        self.base_mean = base_mean
        self.base_variance = base_variance
        
        self.shock_schedule = {}
        self.season_schedule = {i: 1.0 for i in range(7)}
        
        try:
            df_shocks = pd.read_csv(shocks_csv)
            my_shocks = df_shocks[df_shocks['arm_name'] == arm_name]
            self.shock_schedule = dict(zip(my_shocks['step'], my_shocks['multiplier']))
            
            df_season = pd.read_csv(season_csv)
            my_season = df_season[df_season['arm_name'] == arm_name]
            self.season_schedule.update(dict(zip(my_season['day_of_week'], my_season['multiplier'])))
        except FileNotFoundError:
            pass

    def draw(self, t=None):
        if t is None: t = 1
        
        day_of_week = t % 7
        season_multiplier = self.season_schedule.get(day_of_week, 1.0)
        shock_multiplier = self.shock_schedule.get(t, 1.0)
        
        total_multiplier = season_multiplier * shock_multiplier
        current_expected_mean = self.base_mean * total_multiplier
        current_variance = self.base_variance * (total_multiplier ** 2)
        
        current_reward = np.random.normal(current_expected_mean, np.sqrt(abs(current_variance)))
        return max(0.0, current_reward)
import numpy as np
from agents.base_agent import BaseAgent


class ThompsonWeekendWeekday(BaseAgent):
    """주말/평일 분리 Thompson Sampling"""
    def __init__(self, num_arms, reward_scale=7.0, name="Thompson_WeekendWeekday"):
        super().__init__(num_arms, name=name)
        self.reward_scale = reward_scale
        self.alpha = np.ones((2, num_arms))
        self.beta_param = np.ones((2, num_arms))
        self.day = 0

    def _is_weekend(self):
        weekday = self.day % 7
        return weekday >= 5

    def choice(self):
        period = 0 if self._is_weekend() else 1
        samples = np.random.beta(self.alpha[period], self.beta_param[period])
        max_sample = np.max(samples)
        best_arms = np.where(samples == max_sample)[0]
        return np.random.choice(best_arms)

    def getReward(self, arm, reward):
        super().getReward(arm, reward)
        period = 0 if self._is_weekend() else 1
        r = np.clip(reward * self.reward_scale, 0.0, 1.0)
        self.alpha[period, arm] += r
        self.beta_param[period, arm] += (1.0 - r)
        self.day += 1

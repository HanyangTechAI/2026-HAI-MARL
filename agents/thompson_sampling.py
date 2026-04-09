import numpy as np
from agents.base_agent import BaseAgent


class ThompsonSampling(BaseAgent):
    """Thompson Sampling 알고리즘"""
    def __init__(self, num_arms, reward_scale=1.0, window_size=None, name="ThompsonSampling"):
        super().__init__(num_arms, name=name)
        self.reward_scale = reward_scale
        self.alpha = np.ones(num_arms)
        self.beta_param = np.ones(num_arms)
        self.window_size = window_size
        if window_size is not None:
            self.history = []

    def choice(self):
        samples = np.random.beta(self.alpha, self.beta_param)
        max_sample = np.max(samples)
        best_arms = np.where(samples == max_sample)[0]
        return np.random.choice(best_arms)

    def getReward(self, arm, reward):
        super().getReward(arm, reward)
        r = np.clip(reward * self.reward_scale, 0.0, 1.0)
        
        if self.window_size is not None:
            self.history.append((arm, r))
            if len(self.history) > self.window_size:
                self.history.pop(0)
            self.alpha = np.ones(self.num_arms)
            self.beta_param = np.ones(self.num_arms)
            for (a, rw) in self.history:
                self.alpha[a] += rw
                self.beta_param[a] += (1.0 - rw)
        else:
            self.alpha[arm] += r
            self.beta_param[arm] += (1.0 - r)

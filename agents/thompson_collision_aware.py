import numpy as np
from agents.base_agent import BaseAgent


class ThompsonCollisionAware(BaseAgent):
    """충돌 인식 Thompson Sampling"""
    def __init__(self, num_arms, reward_scale=7.0, 
                 collision_penalty_rate=0.5, penalty_decay=0.95,
                 name="Thompson_CollisionAware"):
        super().__init__(num_arms, name=name)
        self.reward_scale = reward_scale
        self.collision_penalty_rate = collision_penalty_rate
        self.penalty_decay = penalty_decay
        self.alpha = np.ones(num_arms)
        self.beta_param = np.ones(num_arms)
        self.collision_penalty = np.ones(num_arms)
        self.expected_reward = np.zeros(num_arms)

    def choice(self):
        samples = np.random.beta(self.alpha, self.beta_param)
        adjusted_samples = samples * self.collision_penalty
        max_sample = np.max(adjusted_samples)
        best_arms = np.where(adjusted_samples == max_sample)[0]
        return np.random.choice(best_arms)

    def getReward(self, arm, reward):
        super().getReward(arm, reward)
        
        if self.pulls[arm] > 1:
            expected = self.q_values[arm]
            if reward < expected * 0.5:
                self.collision_penalty[arm] *= self.collision_penalty_rate
                self.collision_penalty[arm] = max(self.collision_penalty[arm], 0.1)
        
        self.collision_penalty *= self.penalty_decay
        self.collision_penalty = np.minimum(self.collision_penalty, 1.0)
        
        r = np.clip(reward * self.reward_scale, 0.0, 1.0)
        self.alpha[arm] += r
        self.beta_param[arm] += (1.0 - r)

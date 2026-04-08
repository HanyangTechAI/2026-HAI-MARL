import numpy as np
from .base_agent import BaseAgent

class DecayingEpsilonGreedy(BaseAgent):
    """
    시간이 지남에 따라 탐험(Exploration) 확률 epsilon을 점차 줄여나가는 에이전트.
    초반에는 다양한 자산을 탐색하고, 후반부에는 점차 활용(Exploitation)에 집중합니다.
    """
    def __init__(self, num_arms, initial_epsilon=1.0, min_epsilon=0.01, decay_rate=0.99, name="DecayEps"):
        super().__init__(num_arms, name=f"{name}_i{initial_epsilon}_m{min_epsilon}_d{decay_rate}")
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.current_epsilon = initial_epsilon

    def choice(self):
        # 현재 시간에 맞춰 epsilon 감쇠 (지수 감쇠 방식)
        self.current_epsilon = max(self.min_epsilon, self.initial_epsilon * (self.decay_rate ** self.t))
        
        # 1. 탐험 (Exploration): 무작위 선택
        if np.random.rand() < self.current_epsilon:
            return np.random.randint(self.num_arms)
        
        # 2. 활용 (Exploitation): Q값이 가장 높은 자산 선택
        else:
            max_q = np.max(self.q_values)
            best_arms = np.where(self.q_values == max_q)[0]
            return np.random.choice(best_arms)

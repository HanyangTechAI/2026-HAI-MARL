import numpy as np
from .base_agent import BaseAgent

class EpsilonGreedy(BaseAgent):
    """
    가장 기초적인 강화학습 알고리즘: Epsilon-Greedy
    - 확률 epsilon: 새로운 자산을 무작위로 탐색 (Exploration)
    - 확률 1-epsilon: 지금까지 Q-value가 가장 높은 자산을 선택 (Exploitation)
    """
    def __init__(self, num_arms, epsilon=0.1, name="EpsilonGreedy"):
        super().__init__(num_arms, name=f"{name}_e{epsilon}")
        self.epsilon = epsilon

    def choice(self):
        """
        BaseAgent의 choice 함수를 덮어씌움(Override)
        """
        # 1. 탐험 (Exploration): 무작위 선택
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_arms)
        
        # 2. 활용 (Exploitation): Q값이 가장 높은 자산 선택
        else:
            max_q = np.max(self.q_values)
            best_arms = np.where(self.q_values == max_q)[0]
            return np.random.choice(best_arms)
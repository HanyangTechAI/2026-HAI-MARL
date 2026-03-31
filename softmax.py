import numpy as np
from agents.base_agent import BaseAgent

class SoftmaxAgent(BaseAgent):
    """
    Softmax (Boltzmann Exploration) 알고리즘
    - Q-value에 따라 확률적으로 자산을 선택합니다.
    """
    def __init__(self, num_arms, temperature=1.0, name="Softmax"):
        super().__init__(num_arms, name=f"{name}_t{temperature}")
        self.temperature = temperature  # 온도가 높을수록 무작위 탐험, 낮을수록 큰 Q값 극단적 선택

    def choice(self):
        # 오버플로우 방지를 위해 가장 큰 Q값을 빼줌
        q_adjusted = self.q_values - np.max(self.q_values)
        
        # Softmax: e^(Q(a) / T)
        exp_q = np.exp(q_adjusted / self.temperature)
        probs = exp_q / np.sum(exp_q)
        
        # 확률에 따라 자산 선택
        return np.random.choice(self.num_arms, p=probs)

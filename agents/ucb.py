import numpy as np
from agents.base_agent import BaseAgent

class UCBAgent(BaseAgent):
    """
    UCB (Upper Confidence Bound) 알고리즘
    - 탐험과 활용의 균형을 맞추는 알고리즘
    - 선택 횟수가 적은 자산에는 탐험 보너스를 주고, 평균 보상이 높은 자산도 선호합니다.
    """
    def __init__(self, num_arms, c=1.0, name="UCB"):
        super().__init__(num_arms, name=f"{name}_c{c}")
        self.c = c  # 탐험의 정도를 조절하는 하이퍼파라미터

    def choice(self):
        # 1. 한 번도 선택되지 않은 자산이 있다면 우선 탐험
        unexplored = np.where(self.pulls == 0)[0]
        if len(unexplored) > 0:
            return np.random.choice(unexplored)
        
        # 2. 모든 자산이 최소 한 번씩 선택되었다면 UCB 값 계산
        # Q(a) + c * sqrt(ln(t) / N(a))
        ucb_values = self.q_values + self.c * np.sqrt(np.log(self.t) / self.pulls)
        
        # 3. UCB 값이 가장 높은 자산 선택
        max_value = np.max(ucb_values)
        best_arms = np.where(ucb_values == max_value)[0]
        return np.random.choice(best_arms)

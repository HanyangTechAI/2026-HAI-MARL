import numpy as np

class BaseAgent:
    """
    모든 에이전트가 공통으로 상속받을 뼈대 클래스입니다.
    """
    def __init__(self, num_arms, name="BaseAgent"):
        self.num_arms = num_arms
        self.name = name
        
        # -----------------------------------------------------
        # [공통 메모리] 모든 에이전트가 기본적으로 머릿속에 기억해야 할 정보
        # -----------------------------------------------------
        self.pulls = np.zeros(num_arms)      # 각 자산을 몇 번 골랐는지 (N_t)
        self.rewards = np.zeros(num_arms)    # 각 자산에서 얻은 누적 보상 합계
        self.q_values = np.zeros(num_arms)   # 각 자산의 기대 수익률 (Q-value)
        
        self.t = 0  # 현재 에이전트가 체감하는 시간(Step)

    def choice(self):
        """
        어떤 자산을 선택할지 결정하는 행동 로직.
        각 Agent 클래스들은 이 함수를 Override해야 합니다!
        """
        raise NotImplementedError("이 메서드는 하위 클래스에서 반드시 구현해야 합니다.")

    def getReward(self, arm, reward):
        """
        (공통 로직) 심판(Evaluator)이 보상을 던져주면, 메모리(Q-value)를 업데이트합니다.
        가치 기반(Value-based) 강화학습의 가장 핵심이 되는 수식입니다.
        """
        # 1. 고른 횟수와 누적 보상 업데이트
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        
        # 2. Q-value (기대 수익률) 업데이트 수식
        # Q(a) = Q(a) + (1 / N) * [들어온 보상 - Q(a)]
        self.q_values[arm] += (1.0 / self.pulls[arm]) * (reward - self.q_values[arm])
        
        # 3. 시간 1 증가
        self.t += 1
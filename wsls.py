# agents/wsls.py

import numpy as np
from agents.base_agent import BaseAgent

class WSLS(BaseAgent):
    """
    행동경제학 기반 Win-Stay, Lose-Shift 에이전트
    - 기대 수익(Aspiration Level) 이상을 벌면 만족해서 머물고 (Stay)
    - 기대 수익 미만이면 실망해서 다른 랜덤한 곳으로 떠납니다 (Shift)
    """
    def __init__(self, num_arms, name="AI_WSLS", initial_aspiration=0.5, aspiration_lr=0.05):
        # 부모 클래스(BaseAgent)의 초기화 메서드 호출
        super().__init__(num_arms, name=name)
        
        self.initial_aspiration = initial_aspiration 
        self.aspiration_lr = aspiration_lr # 람다(Lambda) 값: 기대치가 현실과 타협하는 속도
        
        self.clear()

    def clear(self):
        self.aspiration_level = self.initial_aspiration
        self.last_action = np.random.randint(self.num_arms)

    def choice(self):
        """다음 스텝의 행동을 결정합니다."""
        # 복잡한 계산 없이, 마음속에 정해진 last_action으로 직진 (근시안적 행동)
        return self.last_action

    def getReward(self, arm, reward):
        """환경으로부터 보상을 받고 기대치를 업데이트합니다."""
        # 1. 감정 평가: Win or Lose?
        if reward >= self.aspiration_level:
            # Win! 만족했으므로 다음에도 여기로 온다. (Stay)
            pass 
        else:
            # Lose! 실망했으므로 다른 곳으로 갈아탄다. (Shift)
            # 현재 매장을 제외한 나머지 매장 중 하나를 무작위로 선택
            possible_arms = [a for a in range(self.num_arms) if a != self.last_action]
            self.last_action = np.random.choice(possible_arms)

        # 2. 현실 타협: 동적 기대 수익 업데이트 (Bendor et al., 2001)
        # New Aspiration = (1 - LR) * Old Aspiration + (LR * Reward)
        self.aspiration_level = (1 - self.aspiration_lr) * self.aspiration_level + (self.aspiration_lr * reward)
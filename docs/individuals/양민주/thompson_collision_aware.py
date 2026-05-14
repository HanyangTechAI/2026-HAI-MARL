import numpy as np
from agents.base_agent import BaseAgent


class ThompsonCollisionAware(BaseAgent):
    """
    충돌 인식 Thompson Sampling
    
    MARL 환경에서 다른 에이전트들과의 충돌을 감지하고 회피하는 Thompson.
    
    핵심 아이디어:
    - 기대 보상보다 실제 보상이 낮으면 "충돌이 많았다"고 판단
    - 충돌이 많은 arm에 패널티를 부여하여 일시적으로 회피
    - 시간이 지나면 패널티가 감소 (다른 에이전트들이 떠났을 수 있음)
    
    Parameters:
        num_arms: arm 개수
        reward_scale: 보상 스케일 배수
        collision_penalty_rate: 충돌 감지 시 패널티 비율 (0~1)
        penalty_decay: 패널티 감소율 (매 턴마다 곱해짐)
        name: 에이전트 이름
    """
    def __init__(self, num_arms, reward_scale=7.0, 
                 collision_penalty_rate=0.5, penalty_decay=0.95,
                 name="Thompson_CollisionAware"):
        super().__init__(num_arms, name=name)
        self.reward_scale = reward_scale
        self.collision_penalty_rate = collision_penalty_rate
        self.penalty_decay = penalty_decay
        
        # Beta 분포
        self.alpha = np.ones(num_arms)
        self.beta_param = np.ones(num_arms)
        
        # 충돌 패널티 (0~1, 낮을수록 회피)
        self.collision_penalty = np.ones(num_arms)
        
        # 기대 보상 추정 (충돌 감지용)
        self.expected_reward = np.zeros(num_arms)

    def choice(self):
        """Beta 샘플링 + 충돌 패널티 적용"""
        samples = np.random.beta(self.alpha, self.beta_param)
        
        # 충돌 패널티 적용 (충돌이 많았던 arm은 샘플 값 감소)
        adjusted_samples = samples * self.collision_penalty
        
        max_sample = np.max(adjusted_samples)
        best_arms = np.where(adjusted_samples == max_sample)[0]
        return np.random.choice(best_arms)

    def getReward(self, arm, reward):
        """보상 수신 + 충돌 감지 + Beta 업데이트"""
        super().getReward(arm, reward)
        
        # 충돌 감지: 실제 보상이 기대보다 많이 낮으면 충돌로 판단
        if self.pulls[arm] > 1:  # 최소 2번은 선택해야 비교 가능
            # 기대 보상 = 현재 Q-value (평균 보상)
            expected = self.q_values[arm]
            
            # 실제 보상이 기대의 50% 이하면 충돌로 판단
            if reward < expected * 0.5:
                # 충돌 패널티 부여 (기존 패널티에 추가로 감소)
                self.collision_penalty[arm] *= self.collision_penalty_rate
                # 최소값 제한 (완전히 0이 되지 않도록)
                self.collision_penalty[arm] = max(self.collision_penalty[arm], 0.1)
        
        # 모든 arm의 패널티를 시간에 따라 회복 (다른 에이전트들이 떠났을 수 있음)
        self.collision_penalty *= self.penalty_decay
        self.collision_penalty = np.minimum(self.collision_penalty, 1.0)  # 최대 1.0
        
        # Beta 분포 업데이트
        r = np.clip(reward * self.reward_scale, 0.0, 1.0)
        self.alpha[arm] += r
        self.beta_param[arm] += (1.0 - r)

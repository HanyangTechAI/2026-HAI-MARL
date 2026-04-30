import numpy as np
from agents.base_agent import BaseAgent


class EXP3(BaseAgent):
    """
    EXP3 (Exponential-weight algorithm for Exploration and Exploitation)
    
    Adversarial(적대적) 환경에 강한 밴딧 알고리즘입니다.
    보상이 확률적이지 않고 최악의 경우(적대적으로 변하는 환경)에도
    이론적으로 보장된 성능(regret bound)을 가집니다.
    
    핵심 아이디어:
      - 각 arm에 "가중치(weight)"를 부여
      - 가중치에 비례한 확률로 arm을 선택하되, 균등 탐험도 섞음
      - 받은 보상을 "선택 확률"로 나눠서 보정 (Importance Weighting)
        → 적게 선택된 arm의 보상이 과소평가되는 것을 방지
      - 보정된 보상으로 가중치를 지수적(exponential)으로 업데이트
    
    비정상 환경 + 슬리피지(충돌 패널티)에 적합한 이유:
      - 보상 분포에 대한 가정이 없어서 환경이 어떻게 변해도 적응
      - 균등 탐험 비율(gamma)이 있어서 특정 arm에 몰리지 않음 → 충돌 회피
      - 확률적 선택이라 다른 에이전트와 행동이 분산됨
    
    Parameters:
        num_arms: arm(선택지)의 개수
        gamma: 탐험 비율 (0~1). 높을수록 균등 탐험, 낮을수록 가중치 기반 활용
        reward_scale: 보상 스케일 배수 (환경 보상이 작을 때 증폭하여 가중치 업데이트 효율 개선)
        name: 에이전트 이름
    """
    def __init__(self, num_arms, gamma=0.1, reward_scale=1.0, name="EXP3"):
        super().__init__(num_arms, name=name)
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.weights = np.ones(num_arms)  # 초기 가중치: 모두 동일

    def _get_probs(self):
        """가중치 기반 선택 확률 계산 (균등 탐험 혼합)"""
        w_sum = np.sum(self.weights)
        probs = (1 - self.gamma) * (self.weights / w_sum) + (self.gamma / self.num_arms)
        return probs

    def choice(self):
        """확률 분포에 따라 arm 선택"""
        probs = self._get_probs()
        return np.random.choice(self.num_arms, p=probs)

    def getReward(self, arm, reward):
        """
        Importance Weighting으로 보상을 보정한 뒤 가중치를 지수적으로 업데이트.
        
        보정 보상 = reward / P(arm) → 적게 선택되는 arm의 보상을 증폭
        가중치 업데이트: w[arm] *= exp(gamma * 보정 보상 / K)
        """
        super().getReward(arm, reward)
        
        probs = self._get_probs()
        
        # 보상을 스케일 보정 후 [0, 1]로 클리핑
        r = np.clip(reward * self.reward_scale, 0.0, 1.0)
        
        # Importance Weighting: 선택 확률로 나눠서 보정
        estimated_reward = r / probs[arm]
        
        # 가중치 지수적 업데이트
        self.weights[arm] *= np.exp(self.gamma * estimated_reward / self.num_arms)
        
        # 오버플로우 방지: 가중치 정규화
        if np.max(self.weights) > 1e10:
            self.weights /= np.max(self.weights)

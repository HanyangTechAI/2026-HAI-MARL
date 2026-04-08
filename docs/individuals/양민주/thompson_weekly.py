import numpy as np
from agents.base_agent import BaseAgent


class ThompsonWeekendWeekday(BaseAgent):
    """
    주말/평일 분리 Thompson Sampling
    
    월마트 데이터는 주말(토,일)과 평일(월~금)의 소비 패턴이 다를 것으로 예상됨.
    주말의 CA_HOBBIES_1과 평일의 CA_HOBBIES_1을 별개로 학습하여
    주기적 패턴을 활용.
    
    Parameters:
        num_arms: arm 개수
        reward_scale: 보상 스케일 배수
        name: 에이전트 이름
    """
    def __init__(self, num_arms, reward_scale=7.0, name="Thompson_WeekendWeekday"):
        super().__init__(num_arms, name=name)
        self.reward_scale = reward_scale
        
        # 주말(0)/평일(1) Beta 분포 유지
        self.alpha = np.ones((2, num_arms))
        self.beta_param = np.ones((2, num_arms))
        
        self.day = 0  # 현재 날짜 (스텝)

    def _is_weekend(self):
        """현재 날짜가 주말인지 판단 (토요일=5, 일요일=6)"""
        weekday = self.day % 7
        return weekday >= 5  # 토요일(5) 또는 일요일(6)

    def choice(self):
        """주말/평일에 따른 Beta 분포에서 샘플링"""
        period = 0 if self._is_weekend() else 1  # 0=주말, 1=평일
        samples = np.random.beta(self.alpha[period], self.beta_param[period])
        
        max_sample = np.max(samples)
        best_arms = np.where(samples == max_sample)[0]
        return np.random.choice(best_arms)

    def getReward(self, arm, reward):
        """주말/평일 Beta 분포만 업데이트"""
        super().getReward(arm, reward)
        
        period = 0 if self._is_weekend() else 1
        r = np.clip(reward * self.reward_scale, 0.0, 1.0)
        
        self.alpha[period, arm] += r
        self.beta_param[period, arm] += (1.0 - r)
        
        self.day += 1

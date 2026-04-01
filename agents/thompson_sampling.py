import numpy as np
from agents.base_agent import BaseAgent


class ThompsonSampling(BaseAgent):
    """
    Thompson Sampling (톰슨 샘플링) 알고리즘
    
    베이지안(Bayesian) 접근법을 사용하는 강화학습 알고리즘입니다.
    각 arm의 보상 분포를 Beta 분포로 모델링하고,
    매 스텝마다 각 arm의 Beta 분포에서 샘플을 뽑아 가장 높은 값의 arm을 선택합니다.
    
    핵심 아이디어:
      - 각 arm마다 "성공(alpha)"과 "실패(beta)" 카운터를 유지
      - 보상이 높으면 alpha 증가, 낮으면 beta 증가
      - 선택 시 Beta(alpha, beta)에서 샘플링 → 자연스럽게 탐험/활용 균형
      - 불확실한 arm은 분산이 커서 가끔 높은 값이 나옴 → 자동 탐험
      - 확실히 좋은 arm은 평균이 높아서 자주 선택됨 → 자동 활용
    
    Parameters:
        num_arms: arm(선택지)의 개수
        window_size: 최근 N스텝만 기억 (비정상 환경 대응). None이면 전체 기억
        name: 에이전트 이름
    """
    def __init__(self, num_arms, window_size=None, name="ThompsonSampling"):
        super().__init__(num_arms, name=name)
        
        # Beta 분포의 파라미터: alpha(성공), beta(실패)
        # 초기값 (1, 1) = 균등 분포 (아무 정보 없는 상태)
        self.alpha = np.ones(num_arms)
        self.beta_param = np.ones(num_arms)
        
        # 슬라이딩 윈도우 (비정상 환경 대응용)
        self.window_size = window_size
        if window_size is not None:
            self.history = []  # (arm, reward) 기록

    def choice(self):
        """
        각 arm의 Beta 분포에서 샘플을 하나씩 뽑고,
        가장 높은 샘플 값을 가진 arm을 선택합니다.
        """
        # 각 arm에서 Beta(alpha, beta) 샘플링
        samples = np.random.beta(self.alpha, self.beta_param)
        
        # 가장 높은 샘플 값의 arm 선택 (동점이면 랜덤)
        max_sample = np.max(samples)
        best_arms = np.where(samples == max_sample)[0]
        return np.random.choice(best_arms)

    def getReward(self, arm, reward):
        """
        보상을 받고 Beta 분포 파라미터를 업데이트합니다.
        
        보상을 [0, 1] 범위의 "성공 확률"로 해석:
          - reward가 높을수록 alpha(성공) 증가
          - reward가 낮을수록 beta(실패) 증가
        """
        # 부모 클래스의 Q-value 업데이트 (공통 통계 유지)
        super().getReward(arm, reward)
        
        # 보상을 [0, 1]로 클리핑 (Beta 분포 요구사항)
        r = np.clip(reward, 0.0, 1.0)
        
        if self.window_size is not None:
            # 슬라이딩 윈도우 모드: 최근 N개만 유지
            self.history.append((arm, r))
            if len(self.history) > self.window_size:
                self.history.pop(0)
            
            # 윈도우 내 데이터로 alpha, beta 재계산
            self.alpha = np.ones(self.num_arms)
            self.beta_param = np.ones(self.num_arms)
            for (a, rw) in self.history:
                self.alpha[a] += rw
                self.beta_param[a] += (1.0 - rw)
        else:
            # 전체 기억 모드: 누적 업데이트
            self.alpha[arm] += r
            self.beta_param[arm] += (1.0 - r)

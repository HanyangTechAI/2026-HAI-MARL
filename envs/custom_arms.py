import numpy as np
from .SMPyBandits.Arms.Arm import Arm

class ShockArm(Arm):
    """
    2) 시장 충격 (Abrupt Change / Changepoint)
    - 평소에는 initial_mean 수익률을 주다가,
    - N번째 스텝(shock_step)이 되는 순간 수익률이 final_mean으로 확 바뀜!
    """
    def __init__(self, initial_mean, final_mean, shock_step, variance=0.05):
        self.initial_mean = initial_mean
        self.final_mean = final_mean
        self.shock_step = shock_step
        self.variance = variance
        self.mean = initial_mean # 초기 평균 수익률

    def draw(self, t=None):
        # 현재 스텝(t)이 충격 시점(shock_step)을 넘었는지 확인
        current_mean = self.final_mean if t >= self.shock_step else self.initial_mean
        return np.random.normal(current_mean, np.sqrt(self.variance))

class TrendArm(Arm):
    """
    3) 추세 상승/하락 (Trending / Periodic)
    - 시간이 지날수록 수익률이 slope만큼 꾸준히 오르거나 떨어지는 주식.
    """
    def __init__(self, start_mean, slope, variance=0.05):
        self.start_mean = start_mean
        self.slope = slope
        self.variance = variance
        self.mean = start_mean

    def draw(self, t=None):
        if t is None: t = 0
        current_mean = self.start_mean + (self.slope * t)
        return np.random.normal(current_mean, np.sqrt(self.variance))

class UniformArm(Arm):
    """
    4) 고변동성 밈 주식 (Uniform Distribution: 0 ~ N)
    - 정규분포가 아니라, 0부터 N(최대값) 사이에서 완전히 무작위로 수익률이 결정됨.
    - 리스크가 극단적으로 높은 자산 모사용.
    """
    def __init__(self, low=0.0, high=1.0):
        self.low = low
        self.high = high
        self.mean = (low + high) / 2.0 # 이론적 평균 기대값

    def draw(self, t=None):
        return np.random.uniform(self.low, self.high)
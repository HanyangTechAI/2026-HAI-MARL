# arms/trend_arm.py
import numpy as np
from .base_arm import BaseArm

class TrendArm(BaseArm):
    """
    시간이 지날수록 수요가 우상향(성장주) 또는 우하향(사양산업)하는 장기 트렌드 자산입니다.
    수식: R(t) = (start_mean + slope * t) + Gaussian_Noise
    """
    def __init__(self, arm_name, start_mean, slope, variance, global_scaler=1.0):
        super().__init__(arm_name=arm_name)
        self.start_mean = start_mean
        self.slope = slope
        self.variance = variance
        self.global_scaler = global_scaler
        
        # 초기 평균값 세팅 (평가기에서 초기 상태 확인할 때 사용)
        self.mean = start_mean * global_scaler

    def draw(self, t=None):
        if t is None: t = 0
        
        # 1. 📈 트렌드 라인 계산: 시간 t에 따른 현재의 '진짜 평균' 수요
        current_expected_value = self.start_mean + (self.slope * t)
        
        # 수요가 마이너스가 되는 것을 방지 (현실에서 판매량이 0 미만이 될 순 없으므로)
        current_expected_value = max(0.0, current_expected_value)
        
        # 2. 노이즈(분산) 추가하여 이번 턴의 실제 수익률 생성
        current_reward = np.random.normal(current_expected_value, np.sqrt(abs(self.variance)))
        
        # 3. 글로벌 스케일링 적용 후 반환
        return current_reward * self.global_scaler
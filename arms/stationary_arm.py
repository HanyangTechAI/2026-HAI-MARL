# arms/stationary_arm.py
import numpy as np
from .base_arm import BaseArm

class StationaryArm(BaseArm):
    """
    안정적인 캐시카우 물류 라인 (변동성이 적고 꾸준한 수요)
    """
    def __init__(self, arm_name, mean, variance):
        super().__init__(arm_name=arm_name)
        self.mean = mean
        self.variance = variance

    def draw(self, t=None):
        """
        주어진 평균과 분산을 바탕으로 정규분포에서 이번 턴의 수익률(수요)을 뽑아냅니다.
        """
        # 분산이 혹시라도 음수가 들어오는 것을 방지하기 위해 max(0, var) 처리
        return np.random.normal(self.mean, np.sqrt(max(0, self.variance)))
# arms/base_arm.py
import numpy as np

class BaseArm:
    """
    모든 커스텀 물류 라인(Arm)의 부모 클래스입니다.
    이 클래스를 상속받는 자식들은 반드시 draw(t) 메서드를 구현해야 합니다.
    """
    def __init__(self, arm_name="Base_Arm"):
        self.arm_name = arm_name
        self.mean = 0.0  # 이론적 평균 (평가기에서 Regret 계산 시 참고용)

    def draw(self, t=None):
        """
        시뮬레이터의 현재 스텝(t)을 입력받아, 이번 턴의 수익률(보상)을 반환합니다.
        자식 클래스에서 오버라이딩(덮어쓰기) 해야 합니다!
        """
        raise NotImplementedError("🚨 에러: 자식 클래스에서 draw(t) 함수를 반드시 구현해야 합니다!")
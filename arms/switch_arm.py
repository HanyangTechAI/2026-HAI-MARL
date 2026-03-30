# arms/switch_arm.py
import numpy as np
import pandas as pd
from .base_arm import BaseArm

class SwitchArm(BaseArm):
    """
    특정 스텝(Day)을 기점으로 기초 평균(Base Mean) 자체가 영구적으로 변하는
    '국면 전환(Regime Change)' 자산 클래스입니다.
    """
    def __init__(self, arm_name, base_mean, base_variance, switch_csv, global_scaler=1.0):
        super().__init__(arm_name=arm_name)
        self.base_mean = base_mean
        self.base_variance = base_variance
        self.global_scaler = global_scaler
        
        # CSV를 읽어 이 Arm에 해당하는 전환 스케줄만 시간순으로 정렬하여 저장
        self.switch_schedule = {}
        try:
            df_switches = pd.read_csv(switch_csv)
            my_switches = df_switches[df_switches['arm_name'] == arm_name]
            
            # 스텝을 오름차순으로 정렬 (예: 150일차에 0.6배, 800일차에 1.2배)
            my_switches = my_switches.sort_values(by='switch_step')
            self.switch_schedule = dict(zip(my_switches['switch_step'], my_switches['multiplier']))
            
        except FileNotFoundError:
            print(f"⚠️ 경고: {switch_csv}를 찾을 수 없습니다. 전환 없이 진행합니다.")
            
        self.mean = base_mean * global_scaler

    def draw(self, t=None):
        if t is None: t = 1

        # 1. 현재 스텝(t)에 맞는 국면(Regime)의 배수 찾기
        current_multiplier = 1.0
        
        # 시간순으로 스케줄을 확인하여, 현재 스텝을 통과한 가장 최근의 배수를 적용
        for step, multiplier in self.switch_schedule.items():
            if t >= step:
                current_multiplier = multiplier
            else:
                break # 아직 도달하지 않은 미래의 전환점은 무시
                
        # 2. 영구적으로 체질이 바뀐 새로운 '현재 평균'
        current_expected_mean = self.base_mean * current_multiplier
        
        # 3. 새로운 평균을 기준으로 노이즈(가우시안) 발생
        # (분산도 변화한 스케일에 맞춰 조정해주면 현실적입니다)
        current_variance = self.base_variance * (current_multiplier ** 2)
        
        current_reward = np.random.normal(current_expected_mean, np.sqrt(abs(current_variance)))
        
        # 4. 수요가 마이너스가 되는 것 방지 및 스케일링
        current_reward = max(0.0, current_reward)
        return current_reward * self.global_scaler
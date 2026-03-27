# arms/event_shock_arm.py
import numpy as np
import pandas as pd
from .base_arm import BaseArm

class EventShockArm(BaseArm):
    """
    월마트의 실제 요일별 주기(Seasonality)와 돌발 충격(Shock)을 
    모두 반영하는 궁극의 Non-stationary 자산 클래스입니다.
    """
    def __init__(self, arm_name, base_mean, base_variance, shocks_csv, season_csv, global_scaler=1.0):
        super().__init__(arm_name=arm_name)
        self.base_mean = base_mean
        self.base_variance = base_variance
        self.global_scaler = global_scaler
        
        # 1. 파동(Seasonality) 데이터 로드 및 딕셔너리 변환
        try:
            df_season = pd.read_csv(season_csv)
            my_season = df_season[df_season['arm_name'] == arm_name]
            # {요일(0~6): 배수(multiplier)} 형태의 딕셔너리로 만듭니다.
            self.season_dict = dict(zip(my_season['day_of_week'], my_season['multiplier']))
        except FileNotFoundError:
            print(f"⚠️ 경고: {season_csv} 파일을 찾을 수 없어 주기성을 1.0으로 초기화합니다.")
            self.season_dict = {}

        # 2. 충격(Shock) 데이터 로드 및 딕셔너리 변환
        try:
            df_shocks = pd.read_csv(shocks_csv)
            my_shocks = df_shocks[df_shocks['arm_name'] == arm_name]
            # {스텝(step): 배수(multiplier)} 형태의 딕셔너리로 만듭니다.
            self.shock_dict = dict(zip(my_shocks['step'], my_shocks['multiplier']))
        except FileNotFoundError:
            print(f"⚠️ 경고: {shocks_csv} 파일을 찾을 수 없어 충격을 발생시키지 않습니다.")
            self.shock_dict = {}
            
        # 평가기(Regret 계산)를 위한 이론적 평균 (초기값)
        self.mean = base_mean * global_scaler

    def draw(self, t=None):
        """
        현재 스텝 t에 맞춰 3단계(노이즈 -> 주기 -> 충격)로 보상을 계산합니다.
        """
        if t is None: t = 1

        # [Step 1] Base 원석 (가우시안 노이즈 포함)
        current_reward = np.random.normal(self.base_mean, np.sqrt(abs(self.base_variance)))

        # [Step 2] 🌊 주기적 파동 (Seasonality) 적용
        # 시작일(d_1)이 토요일이므로 t % 7 을 통해 요일을 구합니다.
        day_of_week = t % 7
        if day_of_week in self.season_dict:
            current_reward *= self.season_dict[day_of_week]

        # [Step 3] ⚡ 돌발 충격 (Shock) 적용
        # 오늘이 혹시 폭증/폭락이 예약된 날짜인지 확인합니다.
        if t in self.shock_dict:
            current_reward *= self.shock_dict[t]
        
        # [Step 4] 글로벌 스케일링 적용 후 최종 반환
        return current_reward * self.global_scaler
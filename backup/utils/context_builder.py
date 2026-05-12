# utils/context_builder.py
import numpy as np
import pandas as pd
import os

class ContextBuilder:
    """
    시점 t와 arm 정보를 받아 contextual feature vector를 생성하는 클래스
    """
    def __init__(self, arm_names):
        self.arm_names = arm_names
        self.num_arms = len(arm_names)
        
        # 데이터 로드
        data_dir = os.path.join("data", "walmart", "extracted_data")
        self.base_params = pd.read_csv(os.path.join(data_dir, "base_params_registry.csv"))
        self.seasonality = pd.read_csv(os.path.join(data_dir, "seasonality_registry.csv"))
        self.shocks = pd.read_csv(os.path.join(data_dir, "shocks_registry.csv"))
        
        # Arm별 파라미터 딕셔너리 생성
        self.arm_params = {}
        for _, row in self.base_params.iterrows():
            self.arm_params[row['arm_name']] = {
                'base_mean': row['base_mean'],
                'slope': row['slope'],
                'base_variance': row['base_variance']
            }
        
        # Arm별 계절성 딕셔너리 생성
        self.seasonality_dict = {}
        for _, row in self.seasonality.iterrows():
            arm = row['arm_name']
            dow = int(row['day_of_week'])
            if arm not in self.seasonality_dict:
                self.seasonality_dict[arm] = {}
            self.seasonality_dict[arm][dow] = row['multiplier']
        
        # Arm별 충격 딕셔너리 생성 (step을 key로)
        self.shocks_dict = {}
        for _, row in self.shocks.iterrows():
            arm = row['arm_name']
            step = int(row['step'])
            if arm not in self.shocks_dict:
                self.shocks_dict[arm] = {}
            self.shocks_dict[arm][step] = row['multiplier']
        
        # Feature dimension 계산
        self.feature_dim = self._calculate_feature_dim()
    
    def _calculate_feature_dim(self):
        """Feature vector의 차원 계산"""
        # 시간 features: 5개
        # - normalized time (1)
        # - day_of_week sin/cos (2)
        # - week sin/cos (2)
        
        # Arm-specific features: 4개
        # - seasonality multiplier (1)
        # - base_mean normalized (1)
        # - slope (1)
        # - base_variance normalized (1)
        
        # Recent shock features: 2개
        # - has_recent_shock (1)
        # - recent_shock_strength (1)
        
        return 11
    
    def get_context(self, t, arm_idx=None):
        """
        시점 t에서의 context feature를 생성
        
        Parameters:
        - t: 현재 시간 스텝 (0 ~ 1940)
        - arm_idx: 특정 arm의 index (None이면 모든 arm의 context 반환)
        
        Returns:
        - context: (feature_dim,) 또는 (num_arms, feature_dim) numpy array
        """
        if arm_idx is not None:
            return self._get_single_context(t, arm_idx)
        else:
            # 모든 arm의 context 반환
            contexts = np.zeros((self.num_arms, self.feature_dim))
            for i in range(self.num_arms):
                contexts[i] = self._get_single_context(t, i)
            return contexts
    
    def _get_single_context(self, t, arm_idx):
        """단일 arm의 context feature 생성"""
        arm_name = self.arm_names[arm_idx]
        features = []
        
        # 1. 시간 정보 (5개)
        normalized_time = t / 1941.0
        day_of_week = t % 7
        week = t // 7
        
        features.append(normalized_time)
        features.append(np.sin(2 * np.pi * day_of_week / 7))
        features.append(np.cos(2 * np.pi * day_of_week / 7))
        features.append(np.sin(2 * np.pi * week / 52))
        features.append(np.cos(2 * np.pi * week / 52))
        
        # 2. 계절성 정보 (1개)
        seasonality_mult = 1.0
        if arm_name in self.seasonality_dict:
            if day_of_week in self.seasonality_dict[arm_name]:
                seasonality_mult = self.seasonality_dict[arm_name][day_of_week]
        features.append(seasonality_mult)
        
        # 3. Arm 고유 특성 (3개)
        if arm_name in self.arm_params:
            params = self.arm_params[arm_name]
            features.append(params['base_mean'] / 10000.0)  # normalized
            features.append(params['slope'])
            features.append(params['base_variance'] / 100000.0)  # normalized
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # 4. 최근 충격 정보 (2개)
        has_shock, shock_strength = self._get_recent_shock_info(arm_name, t, window=7)
        features.append(float(has_shock))
        features.append(shock_strength)
        
        return np.array(features)
    
    def _get_recent_shock_info(self, arm_name, t, window=7):
        """최근 window 일 내의 충격 정보 반환"""
        if arm_name not in self.shocks_dict:
            return False, 0.0
        
        # 최근 window 일 내에 충격이 있었는지 확인
        for past_t in range(max(0, t - window), t + 1):
            if past_t in self.shocks_dict[arm_name]:
                shock_mult = self.shocks_dict[arm_name][past_t]
                # 충격 강도를 -1 ~ 1 범위로 정규화
                # multiplier가 1.0이면 0, 2.0이면 1, 0.0이면 -1
                normalized_strength = shock_mult - 1.0
                return True, normalized_strength
        
        return False, 0.0

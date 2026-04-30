# docs/individuals/양민주/lin_ucb.py
"""
LinUCB (Linear Upper Confidence Bound) Algorithm

Contextual Bandit 알고리즘으로, Ridge Regression을 사용하여 
각 arm의 보상을 예측하고 UCB 방식으로 탐색-활용 균형을 맞춥니다.

참고 논문:
- Li et al. (2010) "A Contextual-Bandit Approach to Personalized News Article Recommendation"
"""

import numpy as np
from agents.base_agent import BaseAgent

class LinUCB(BaseAgent):
    """
    Linear Upper Confidence Bound with Ridge Regression
    
    각 arm에 대해 독립적인 선형 모델을 학습하며,
    context feature를 활용하여 보상을 예측합니다.
    """
    
    def __init__(self, num_arms, feature_dim, alpha=1.0, lambda_reg=1.0, name="LinUCB"):
        """
        Parameters:
        - num_arms: arm의 개수
        - feature_dim: context feature vector의 차원
        - alpha: exploration parameter (클수록 더 탐색적)
        - lambda_reg: L2 regularization parameter (Ridge regression)
        - name: 에이전트 이름
        """
        super().__init__(num_arms, name=name)
        
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # Per-arm parameters
        # A_a: (d x d) 행렬, context의 outer product 누적
        # b_a: (d,) 벡터, context * reward 누적
        self.A = [np.eye(feature_dim) * lambda_reg for _ in range(num_arms)]
        self.b = [np.zeros(feature_dim) for _ in range(num_arms)]
        
        # Context를 저장할 변수 (외부에서 설정)
        self.current_context = None
    
    def set_context(self, context):
        """
        현재 시점의 context를 설정
        
        Parameters:
        - context: (num_arms, feature_dim) numpy array
        """
        self.current_context = context
    
    def choice(self):
        """
        UCB 값이 가장 높은 arm을 선택
        
        Returns:
        - arm_idx: 선택된 arm의 index
        """
        if self.current_context is None:
            # Context가 없으면 랜덤 선택
            return np.random.randint(0, self.num_arms)
        
        ucb_values = np.zeros(self.num_arms)
        
        for arm in range(self.num_arms):
            x = self.current_context[arm]  # (feature_dim,)
            
            # Ridge regression solution: theta = A^{-1} * b
            A_inv = np.linalg.inv(self.A[arm])
            theta = A_inv @ self.b[arm]
            
            # 예측 보상
            predicted_reward = theta @ x
            
            # 불확실성 (confidence bound)
            uncertainty = self.alpha * np.sqrt(x @ A_inv @ x)
            
            # UCB = 예측값 + 불확실성
            ucb_values[arm] = predicted_reward + uncertainty
        
        return np.argmax(ucb_values)
    
    def getReward(self, arm, reward):
        """
        관찰된 보상으로 모델 업데이트
        
        Parameters:
        - arm: 선택된 arm의 index
        - reward: 관찰된 보상
        """
        if self.current_context is None:
            # Context가 없으면 업데이트 불가
            super().getReward(arm, reward)
            return
        
        x = self.current_context[arm]  # (feature_dim,)
        
        # A_a 업데이트: A_a = A_a + x * x^T
        self.A[arm] += np.outer(x, x)
        
        # b_a 업데이트: b_a = b_a + r * x
        self.b[arm] += reward * x
        
        # 기본 통계 업데이트 (BaseAgent)
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        self.q_values[arm] = self.rewards[arm] / self.pulls[arm] if self.pulls[arm] > 0 else 0
        self.t += 1
    
    def get_theta(self, arm):
        """
        특정 arm의 학습된 파라미터 theta 반환 (디버깅용)
        
        Parameters:
        - arm: arm index
        
        Returns:
        - theta: (feature_dim,) numpy array
        """
        A_inv = np.linalg.inv(self.A[arm])
        return A_inv @ self.b[arm]
    
    def get_feature_importance(self, arm):
        """
        특정 arm의 feature 중요도 반환 (theta의 절댓값)
        
        Parameters:
        - arm: arm index
        
        Returns:
        - importance: (feature_dim,) numpy array
        """
        theta = self.get_theta(arm)
        return np.abs(theta)

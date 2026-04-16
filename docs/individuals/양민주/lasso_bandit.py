# docs/individuals/양민주/lasso_bandit.py
"""
Lasso Bandit Algorithm

Contextual Bandit 알고리즘으로, Lasso Regression (L1 regularization)을 사용하여
중요한 feature만 선택하면서 각 arm의 보상을 예측합니다.

참고 논문:
- Bastani & Bayati (2020) "Online Decision Making with High-Dimensional Covariates"
"""

import numpy as np
from agents.base_agent import BaseAgent

class LassoBandit(BaseAgent):
    """
    Lasso Bandit with L1 Regularization
    
    L1 regularization을 통해 feature selection을 수행하며,
    중요한 feature만 사용하여 보상을 예측합니다.
    
    Note: 완전한 Lasso는 coordinate descent가 필요하지만,
    여기서는 간소화된 버전으로 elastic net (L1 + L2)을 사용합니다.
    """
    
    def __init__(self, num_arms, feature_dim, alpha=1.0, lambda_l1=0.1, lambda_l2=1.0, 
                 learning_rate=0.01, name="LassoBandit"):
        """
        Parameters:
        - num_arms: arm의 개수
        - feature_dim: context feature vector의 차원
        - alpha: exploration parameter (클수록 더 탐색적)
        - lambda_l1: L1 regularization parameter (feature selection)
        - lambda_l2: L2 regularization parameter (stability)
        - learning_rate: gradient descent learning rate
        - name: 에이전트 이름
        """
        super().__init__(num_arms, name=name)
        
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.learning_rate = learning_rate
        
        # Per-arm parameters
        # theta: (d,) 벡터, 학습된 가중치
        self.theta = [np.zeros(feature_dim) for _ in range(num_arms)]
        
        # 각 arm의 context와 reward 히스토리 저장
        self.context_history = [[] for _ in range(num_arms)]
        self.reward_history = [[] for _ in range(num_arms)]
        
        # Confidence bound 계산을 위한 정보
        self.A = [np.eye(feature_dim) * lambda_l2 for _ in range(num_arms)]
        
        # Context를 저장할 변수 (외부에서 설정)
        self.current_context = None
        
        # 업데이트 주기 (매 update_freq번마다 theta 재학습)
        self.update_freq = 10
        self.update_counter = [0 for _ in range(num_arms)]
    
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
            
            # 예측 보상
            predicted_reward = self.theta[arm] @ x
            
            # 불확실성 (confidence bound)
            A_inv = np.linalg.inv(self.A[arm])
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
        
        # 히스토리에 추가
        self.context_history[arm].append(x)
        self.reward_history[arm].append(reward)
        
        # A 행렬 업데이트 (confidence bound 계산용)
        self.A[arm] += np.outer(x, x)
        
        # 주기적으로 theta 재학습
        self.update_counter[arm] += 1
        if self.update_counter[arm] >= self.update_freq:
            self._update_theta(arm)
            self.update_counter[arm] = 0
        else:
            # 간단한 gradient descent 업데이트
            self._gradient_update(arm, x, reward)
        
        # 기본 통계 업데이트 (BaseAgent)
        self.pulls[arm] += 1
        self.rewards[arm] += reward
        self.q_values[arm] = self.rewards[arm] / self.pulls[arm] if self.pulls[arm] > 0 else 0
        self.t += 1
    
    def _gradient_update(self, arm, x, reward):
        """
        단일 샘플에 대한 gradient descent 업데이트
        
        Parameters:
        - arm: arm index
        - x: context vector
        - reward: observed reward
        """
        # 예측 오차
        prediction = self.theta[arm] @ x
        error = reward - prediction
        
        # Gradient descent with elastic net regularization
        # theta = theta + lr * (error * x - lambda_l2 * theta - lambda_l1 * sign(theta))
        gradient = error * x - self.lambda_l2 * self.theta[arm]
        
        # L1 regularization (soft thresholding)
        self.theta[arm] += self.learning_rate * gradient
        self.theta[arm] = self._soft_threshold(self.theta[arm], 
                                                self.learning_rate * self.lambda_l1)
    
    def _update_theta(self, arm):
        """
        전체 히스토리를 사용하여 theta를 재학습 (Elastic Net)
        
        Parameters:
        - arm: arm index
        """
        if len(self.context_history[arm]) == 0:
            return
        
        X = np.array(self.context_history[arm])  # (n_samples, feature_dim)
        y = np.array(self.reward_history[arm])   # (n_samples,)
        
        # Coordinate descent for elastic net (간소화 버전)
        # 실제로는 sklearn의 ElasticNet을 사용하는 것이 더 정확하지만,
        # 여기서는 의존성을 줄이기 위해 간단한 버전 사용
        
        n_iterations = 20
        for _ in range(n_iterations):
            # 예측
            predictions = X @ self.theta[arm]
            errors = y - predictions
            
            # Gradient
            gradient = (X.T @ errors) / len(y) - self.lambda_l2 * self.theta[arm]
            
            # Update with soft thresholding
            self.theta[arm] += self.learning_rate * gradient
            self.theta[arm] = self._soft_threshold(self.theta[arm], 
                                                    self.learning_rate * self.lambda_l1)
    
    def _soft_threshold(self, x, threshold):
        """
        Soft thresholding operator for L1 regularization
        
        Parameters:
        - x: input array
        - threshold: threshold value
        
        Returns:
        - thresholded array
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
    
    def get_theta(self, arm):
        """
        특정 arm의 학습된 파라미터 theta 반환 (디버깅용)
        
        Parameters:
        - arm: arm index
        
        Returns:
        - theta: (feature_dim,) numpy array
        """
        return self.theta[arm].copy()
    
    def get_feature_importance(self, arm):
        """
        특정 arm의 feature 중요도 반환 (theta의 절댓값)
        
        Parameters:
        - arm: arm index
        
        Returns:
        - importance: (feature_dim,) numpy array
        """
        return np.abs(self.theta[arm])
    
    def get_selected_features(self, arm, threshold=1e-4):
        """
        L1 regularization에 의해 선택된 feature 반환
        
        Parameters:
        - arm: arm index
        - threshold: feature를 선택하는 임계값
        
        Returns:
        - selected_indices: 선택된 feature의 index 리스트
        """
        importance = self.get_feature_importance(arm)
        return np.where(importance > threshold)[0].tolist()

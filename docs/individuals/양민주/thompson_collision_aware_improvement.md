# Thompson Collision Aware 알고리즘 개선 방안

## 📊 현재 알고리즘 분석

### 핵심 메커니즘

```python
class ThompsonCollisionAware:
    def __init__(self, reward_scale=7.0, collision_penalty_rate=0.5, penalty_decay=0.95):
        self.collision_penalty = np.ones(num_arms)  # 충돌 페널티 (초기값 1.0)
        
    def choice(self):
        samples = np.random.beta(self.alpha, self.beta_param)
        adjusted_samples = samples * self.collision_penalty  # 페널티 적용
        return argmax(adjusted_samples)
    
    def getReward(self, arm, reward):
        # 충돌 감지: 보상이 기대값의 50% 미만이면
        if reward < expected * 0.5:
            self.collision_penalty[arm] *= 0.5  # 페널티 강화
        
        # 모든 arm의 페널티를 0.95배로 감소 (회복)
        self.collision_penalty *= 0.95
```

### 강점
1. ✅ **충돌 감지**: 보상이 급격히 낮아지면 충돌로 판단
2. ✅ **동적 조정**: 페널티가 시간에 따라 회복됨
3. ✅ **다중 에이전트 환경 고려**: 슬리피지 문제 인식

### 약점
1. ❌ **고정된 임계값**: `expected * 0.5`가 모든 상황에 적합하지 않을 수 있음
2. ❌ **단순한 페널티 메커니즘**: 충돌 강도를 고려하지 않음
3. ❌ **전역 decay**: 선택되지 않은 arm도 페널티가 회복됨
4. ❌ **충돌 이력 미활용**: 과거 충돌 패턴을 학습하지 않음

---

## 🚀 개선 아이디어

### 1. **적응형 충돌 임계값 (Adaptive Collision Threshold)**

**문제**: 현재 `expected * 0.5`는 고정값
**해결**: 보상의 변동성을 고려한 동적 임계값

```python
class ImprovedThompsonCollisionAware:
    def __init__(self, num_arms, threshold_sigma=2.0):
        self.threshold_sigma = threshold_sigma  # 표준편차 배수
        self.reward_variance = np.zeros(num_arms)  # 보상 분산 추적
        
    def getReward(self, arm, reward):
        # 분산 업데이트 (온라인 알고리즘)
        n = self.pulls[arm]
        old_mean = self.q_values[arm]
        new_mean = old_mean + (reward - old_mean) / n
        self.reward_variance[arm] += (reward - old_mean) * (reward - new_mean)
        
        # 동적 임계값: mean - threshold_sigma * std
        std = np.sqrt(self.reward_variance[arm] / n) if n > 1 else 0
        threshold = self.q_values[arm] - self.threshold_sigma * std
        
        # 충돌 감지
        if reward < threshold:
            collision_severity = (threshold - reward) / (threshold + 1e-6)
            self.collision_penalty[arm] *= (1 - collision_severity * 0.5)
```

**장점**:
- 각 arm의 변동성을 고려
- 안정적인 arm은 엄격한 기준, 불안정한 arm은 관대한 기준
- 충돌 강도에 비례한 페널티

---

### 2. **충돌 이력 기반 학습 (Collision History Learning)**

**문제**: 과거 충돌 패턴을 활용하지 않음
**해결**: 충돌 빈도와 패턴을 학습

```python
class CollisionHistoryThompson:
    def __init__(self, num_arms, history_window=100):
        self.collision_count = np.zeros(num_arms)  # 충돌 횟수
        self.collision_history = [[] for _ in range(num_arms)]  # 시간별 충돌 기록
        self.history_window = history_window
        
    def getReward(self, arm, reward):
        is_collision = reward < self.q_values[arm] * 0.5
        
        # 충돌 이력 업데이트
        self.collision_history[arm].append((self.t, is_collision))
        if len(self.collision_history[arm]) > self.history_window:
            self.collision_history[arm].pop(0)
        
        # 최근 충돌 비율 계산
        recent_collisions = sum(1 for _, col in self.collision_history[arm] if col)
        collision_rate = recent_collisions / len(self.collision_history[arm])
        
        # 충돌 비율에 따른 페널티 조정
        self.collision_penalty[arm] = 1.0 - (collision_rate * 0.7)  # 최대 70% 감소
```

**장점**:
- 지속적으로 충돌이 많은 arm을 회피
- 일시적 충돌과 구조적 충돌 구분
- 슬라이딩 윈도우로 최근 패턴 반영

---

### 3. **시간대별 충돌 패턴 학습 (Temporal Collision Pattern)**

**문제**: 특정 시간대에 충돌이 많을 수 있음 (예: 주말)
**해결**: 요일/시간대별 충돌 확률 학습

```python
class TemporalCollisionThompson:
    def __init__(self, num_arms, period=7):
        self.period = period  # 주기 (7 = 주간)
        # arm별, 시간대별 충돌 확률
        self.collision_prob = np.ones((num_arms, period)) * 0.5
        self.time_counts = np.zeros((num_arms, period))
        
    def choice(self):
        time_slot = self.t % self.period
        samples = np.random.beta(self.alpha, self.beta_param)
        
        # 현재 시간대의 충돌 확률로 조정
        collision_risk = self.collision_prob[:, time_slot]
        adjusted_samples = samples * (1 - collision_risk * 0.5)
        
        return np.argmax(adjusted_samples)
    
    def getReward(self, arm, reward):
        time_slot = self.t % self.period
        is_collision = reward < self.q_values[arm] * 0.5
        
        # 시간대별 충돌 확률 업데이트 (지수 이동 평균)
        alpha = 0.1
        self.collision_prob[arm, time_slot] = (
            (1 - alpha) * self.collision_prob[arm, time_slot] + 
            alpha * float(is_collision)
        )
```

**장점**:
- 주기적 패턴 활용 (주말 vs 평일)
- 시간대별 최적 전략 학습
- Walmart 데이터의 계절성 활용

---

### 4. **다중 에이전트 인식 (Multi-Agent Awareness)**

**문제**: 다른 에이전트의 선택을 고려하지 않음
**해결**: 인기 있는 arm을 회피

```python
class MultiAgentAwareThompson:
    def __init__(self, num_arms, num_agents=8):
        self.num_agents = num_agents
        self.arm_popularity = np.zeros(num_arms)  # arm별 인기도
        self.popularity_decay = 0.9
        
    def choice(self):
        samples = np.random.beta(self.alpha, self.beta_param)
        
        # 인기도 기반 페널티 (많이 선택될수록 페널티)
        # 예상 충돌 수 = popularity * (num_agents - 1)
        expected_collisions = self.arm_popularity * (self.num_agents - 1)
        collision_penalty = 1.0 / (1.0 + expected_collisions * 0.3)
        
        adjusted_samples = samples * collision_penalty
        return np.argmax(adjusted_samples)
    
    def getReward(self, arm, reward):
        # 보상이 낮으면 해당 arm이 인기 있었다고 추정
        if reward < self.q_values[arm] * 0.5:
            self.arm_popularity[arm] += 0.5
        
        # 인기도 감소 (시간에 따라)
        self.arm_popularity *= self.popularity_decay
        self.arm_popularity = np.clip(self.arm_popularity, 0, 5)
```

**장점**:
- 다른 에이전트와의 암묵적 협력
- 혼잡한 arm 회피
- 슬리피지 최소화

---

### 5. **하이브리드: UCB + Thompson Sampling**

**문제**: Thompson Sampling은 순수 확률적
**해결**: UCB의 불확실성 개념 통합

```python
class HybridThompsonUCB:
    def __init__(self, num_arms, ucb_weight=0.3, c=0.1):
        self.ucb_weight = ucb_weight  # UCB 가중치
        self.c = c  # UCB exploration parameter
        
    def choice(self):
        # Thompson Sampling 샘플
        thompson_samples = np.random.beta(self.alpha, self.beta_param)
        
        # UCB 값 계산
        ucb_values = self.q_values + self.c * np.sqrt(
            np.log(self.t + 1) / (self.pulls + 1)
        )
        
        # 정규화 후 결합
        thompson_norm = thompson_samples / (np.max(thompson_samples) + 1e-6)
        ucb_norm = ucb_values / (np.max(ucb_values) + 1e-6)
        
        combined = (1 - self.ucb_weight) * thompson_norm + self.ucb_weight * ucb_norm
        
        # 충돌 페널티 적용
        adjusted = combined * self.collision_penalty
        
        return np.argmax(adjusted)
```

**장점**:
- Thompson의 탐색 + UCB의 체계적 불확실성 관리
- 더 안정적인 성능
- 파라미터로 균형 조절 가능

---

### 6. **적응형 Reward Scale**

**문제**: `reward_scale=7.0`이 고정값
**해결**: 환경에 따라 자동 조정

```python
class AdaptiveScaleThompson:
    def __init__(self, num_arms):
        self.reward_scale = 1.0  # 초기값
        self.reward_history = []
        self.scale_update_freq = 100
        
    def getReward(self, arm, reward):
        self.reward_history.append(reward)
        
        # 주기적으로 scale 조정
        if len(self.reward_history) % self.scale_update_freq == 0:
            # 보상 범위 분석
            rewards = np.array(self.reward_history[-self.scale_update_freq:])
            reward_range = np.percentile(rewards, 95) - np.percentile(rewards, 5)
            
            # Scale 조정: 보상을 [0, 1] 범위로 매핑
            if reward_range > 0:
                self.reward_scale = 1.0 / reward_range
            
        # 조정된 scale 사용
        r = np.clip(reward * self.reward_scale, 0.0, 1.0)
        self.alpha[arm] += r
        self.beta_param[arm] += (1.0 - r)
```

**장점**:
- 환경 변화에 자동 적응
- 수동 튜닝 불필요
- 다양한 보상 분포에 강건

---

## 🎯 추천 개선 조합

### **조합 1: 기본 개선 (쉬움)**
```python
ThompsonCollisionAware(
    reward_scale=10.0,              # 7.0 → 10.0 (더 민감한 학습)
    collision_penalty_rate=0.3,     # 0.5 → 0.3 (덜 공격적인 페널티)
    penalty_decay=0.98              # 0.95 → 0.98 (더 느린 회복)
)
```

### **조합 2: 적응형 임계값 (중간)**
- 아이디어 1 (적응형 충돌 임계값) 구현
- `threshold_sigma=1.5` ~ `2.5` 테스트

### **조합 3: 시간 패턴 학습 (고급)**
- 아이디어 3 (시간대별 충돌 패턴) 구현
- Walmart 데이터의 주간 패턴 활용
- `period=7` (주간 주기)

### **조합 4: 하이브리드 (최고급)**
- 아이디어 5 (UCB + Thompson) 구현
- `ucb_weight=0.2` ~ `0.4` 테스트
- 안정성과 탐색의 균형

---

## 📊 실험 계획

### Phase 1: 파라미터 튜닝
```python
# 현재 파라미터
reward_scale: 7.0
collision_penalty_rate: 0.5
penalty_decay: 0.95

# 테스트할 조합
configs = [
    (5.0, 0.3, 0.95),
    (7.0, 0.3, 0.98),
    (10.0, 0.3, 0.98),
    (10.0, 0.4, 0.97),
    (15.0, 0.4, 0.98),
]
```

### Phase 2: 적응형 임계값
```python
threshold_sigma: [1.0, 1.5, 2.0, 2.5, 3.0]
```

### Phase 3: 시간 패턴
```python
period: [7, 14, 30]  # 주간, 격주, 월간
```

### Phase 4: 하이브리드
```python
ucb_weight: [0.1, 0.2, 0.3, 0.4, 0.5]
```

---

## 💡 예상 성능 향상

### 현재 성능 (추정)
- Thompson Collision Aware: 중상위권 (정확한 수치 필요)

### 개선 후 목표
1. **파라미터 튜닝**: +5~10% 성능 향상
2. **적응형 임계값**: +10~15% 성능 향상
3. **시간 패턴 학습**: +15~20% 성능 향상 (Walmart 데이터에 특히 효과적)
4. **하이브리드**: +20~30% 성능 향상 (가장 안정적)

### 목표
- **Epsilon Greedy (0.05)를 넘어서기**: 113.87 이상
- **UCB (0.1)과 경쟁**: 107.63 수준

---

## 🔧 구현 우선순위

### 1순위: 파라미터 튜닝 (즉시 가능)
- 코드 수정 최소
- 빠른 실험 가능
- 기본 성능 향상

### 2순위: 적응형 임계값 (1~2시간)
- 구현 난이도 낮음
- 명확한 개선 효과
- 범용성 높음

### 3순위: 시간 패턴 학습 (2~3시간)
- Walmart 데이터에 최적화
- 계절성 활용
- 실용적 가치 높음

### 4순위: 하이브리드 (3~4시간)
- 구현 복잡도 높음
- 최고 성능 기대
- 연구 가치 높음

---

## 📝 다음 단계

1. ✅ **현재 Thompson Collision Aware 성능 정확히 측정**
2. ✅ **파라미터 Grid Search 실행**
3. ✅ **적응형 임계값 구현 및 테스트**
4. ✅ **시간 패턴 학습 구현 및 테스트**
5. ✅ **최종 성능 비교 및 분석**

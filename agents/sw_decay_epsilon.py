import numpy as np
from collections import deque
from .base_agent import BaseAgent


class SWDecayEpsilonGreedy(BaseAgent):
    """
    Sliding Window + Decaying Epsilon Greedy 하이브리드 에이전트
    ============================================================

    핵심 아이디어:
      18-Agent 극한 경쟁 실험에서 1위(DecayEps)와 2위(SW-UCB)의 강점을 결합.

    두 알고리즘의 역할 분담:
      [Decaying Epsilon] — "언제" 탐색할지 결정
        - 초반엔 높은 ε으로 적극 탐색 (Exploration)
        - 시간이 지나면서 ε이 지수 감쇠 → Exploitation으로 자연스럽게 전환
        - 극한 경쟁(18명)에서 가장 안정적임이 입증됨

      [Sliding Window] — "무엇을" 선택할지 결정
        - Q-value를 전체 평균 대신 최근 window 내 평균으로 계산
        - 트렌드 전환(SwitchArm)과 이벤트 충격(EventShockArm)에 빠르게 반응
        - 오래된 정보를 잊어버림 (Forgetting Mechanism)

    탐색 모드 (ε 발동):
      - 완전한 무작위 대신, 각 Arm의 sliding window 평균을
        Softmax 온도로 스케일링한 확률로 "편향된 탐색"
      - 결국 좋아 보이는 Arm 쪽으로 탐색이 쏠리되, 불확실성도 유지

    착취 모드 (1-ε 발동):
      - window 내 평균(q_w)을 Q-value로 사용하여 argmax 선택
      - 최근 데이터만 보기 때문에 환경 변화에 민감하게 반응

    파라미터:
      num_arms        : 선택 가능한 Arm 수
      window_size     : 슬라이딩 윈도우 크기 (최근 N번의 기록만 유지)
      initial_epsilon : 초기 탐색 확률 (기본 0.8)
      min_epsilon     : 최소 탐색 확률 (이 이하로는 내려가지 않음)
      decay_rate      : epsilon 지수 감쇠 계수 (0 < rate < 1)
      explore_temp    : 탐색 시 Softmax 온도 (높을수록 더 균등한 탐색)
    """

    def __init__(
        self,
        num_arms: int,
        window_size: int = 200,
        initial_epsilon: float = 0.8,
        min_epsilon: float = 0.01,
        decay_rate: float = 0.995,
        explore_temp: float = 0.3,
        name: str = "SW_DecayEps",
    ):
        super().__init__(num_arms, name=name)
        self.window_size = window_size
        self.initial_epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.explore_temp = explore_temp

        # 각 Arm별 최근 보상 기록 (deque는 maxlen 초과 시 자동으로 오래된 것 삭제)
        self.window: list[deque] = [
            deque(maxlen=window_size) for _ in range(num_arms)
        ]
        self.current_epsilon = initial_epsilon

    def _window_q(self, arm: int) -> tuple[float, int]:
        """
        슬라이딩 윈도우 내의 평균 보상과 샘플 수를 반환.
        window가 비어 있으면 (0.0, 0) 반환.
        """
        if len(self.window[arm]) == 0:
            return 0.0, 0
        rewards = [r for r in self.window[arm]]
        return float(np.mean(rewards)), len(rewards)

    def _window_q_values(self) -> np.ndarray:
        """
        모든 Arm에 대한 슬라이딩 윈도우 Q-value 배열을 반환.
        비어 있는 Arm은 0.0으로 처리.
        """
        q_w = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            q_w[arm], _ = self._window_q(arm)
        return q_w

    def choice(self) -> int:
        # 1. Epsilon 지수 감쇠 계산 (매 스텝마다 업데이트)
        self.current_epsilon = max(
            self.min_epsilon,
            self.initial_epsilon * (self.decay_rate ** self.t)
        )

        # 2. 아직 한 번도 선택 안 된 Arm 우선 탐색 (초기 탐색 보장)
        unexplored = [a for a in range(self.num_arms) if len(self.window[a]) == 0]
        if unexplored:
            return int(np.random.choice(unexplored))

        # 슬라이딩 윈도우 Q-value 계산
        q_w = self._window_q_values()

        # 3. Epsilon 발동 여부 결정
        if np.random.rand() < self.current_epsilon:
            # ── 탐색 모드 (Exploration) ──────────────────────────
            # 완전 무작위 대신 sliding window Q-value 기반 Softmax 확률로 편향 탐색.
            # explore_temp가 낮을수록 좋아 보이는 Arm 쪽으로 탐색이 쏠림.
            # explore_temp가 높을수록 균등한 무작위에 가까워짐.
            scaled = q_w / self.explore_temp
            # 수치 안정성을 위해 max를 빼줌 (softmax shift trick)
            scaled -= scaled.max()
            probs = np.exp(scaled)
            probs /= probs.sum()
            return int(np.random.choice(self.num_arms, p=probs))
        else:
            # ── 착취 모드 (Exploitation) ──────────────────────────
            # 최근 window 내 평균이 가장 높은 Arm 선택.
            # 동점 시 무작위로 하나 선택 (탐색 다양성 유지).
            max_q = np.max(q_w)
            best_arms = np.where(q_w == max_q)[0]
            return int(np.random.choice(best_arms))

    def getReward(self, arm: int, reward: float):
        # BaseAgent의 공통 업데이트 (pulls, rewards, q_values, t)
        super().getReward(arm, reward)
        # Sliding Window에 최신 보상 추가 (maxlen 초과 시 자동으로 가장 오래된 것 제거)
        self.window[arm].append(reward)

import numpy as np
from .base_agent import BaseAgent


class PeriodicUCB(BaseAgent):
    """
    Periodic UCB 알고리즘

    기존 UCB에 삼각함수 기반 주기성을 두 가지 방식으로 결합합니다.

    1) 수식에 sin 탐험 보너스 추가 (주기적 탐험 강화)
    2) 탐험 강도 c를 시간에 따라 진동 (주기적 c 스케일링)

    선택 기준:
      UCB_base(a)  = Q(a) + c_t * sqrt(ln(t) / N(a))
      periodic(a)  = amplitude * sin(2π * t / period + phase_offset * a)
      score(a)     = UCB_base(a) + periodic(a)

      c_t = c_base * (1 + c_amplitude * sin(2π * t / c_period))

    Args:
        num_arms      : arm 개수
        c             : 기본 탐험 강도 (기본값 0.1)
        period        : 보상 주기성 감지 주기 (기본값 7, 주간 패턴)
        amplitude     : sin 탐험 보너스 크기 (기본값 0.05)
        phase_offset  : arm 간 위상 오프셋 — arm마다 다른 타이밍에 탐험
                        (기본값 0 → 모든 arm 동위상 / 0이 아니면 arm별로 2π/num_arms씩 오프셋)
        c_period      : c 진동 주기 (기본값 period와 동일)
        c_amplitude   : c 진동 폭 (기본값 0.3 → c가 ±30% 진동)
        name          : 에이전트 이름
    """

    def __init__(
        self,
        num_arms: int,
        c: float = 0.1,
        period: float = 7.0,
        amplitude: float = 0.05,
        phase_offset: float = 0.0,
        c_period: float = None,
        c_amplitude: float = 0.3,
        name: str = "PeriodicUCB",
    ):
        super().__init__(num_arms, name=name)
        self.c           = c
        self.period      = period
        self.amplitude   = amplitude
        self.phase_offset = phase_offset
        self.c_period    = c_period if c_period is not None else period
        self.c_amplitude = c_amplitude

        # arm별 위상 오프셋 벡터
        # phase_offset=0 → 모두 동위상
        # phase_offset=1 → arm마다 2π/num_arms 씩 어긋남
        self._arm_phases = np.array([
            phase_offset * (2 * np.pi / num_arms) * i
            for i in range(num_arms)
        ])

    # ------------------------------------------------------------------
    # 주기 관련 계산
    # ------------------------------------------------------------------

    def _c_t(self) -> float:
        """시간에 따라 진동하는 탐험 강도 c_t"""
        oscillation = self.c_amplitude * np.sin(2 * np.pi * self.t / self.c_period)
        return self.c * (1.0 + oscillation)

    def _periodic_bonus(self) -> np.ndarray:
        """각 arm에 대한 sin 기반 탐험 보너스 벡터"""
        phase = 2 * np.pi * self.t / self.period + self._arm_phases
        return self.amplitude * np.sin(phase)

    # ------------------------------------------------------------------
    # BaseAgent 인터페이스 구현
    # ------------------------------------------------------------------

    def choice(self) -> int:
        """
        1. 미탐험 arm 우선 선택 (기존 UCB와 동일)
        2. UCB 기본값 + 주기 보너스로 최종 점수 계산
        """
        # 미탐험 arm 우선
        unexplored = np.where(self.pulls == 0)[0]
        if len(unexplored) > 0:
            return int(np.random.choice(unexplored))

        # c_t: 시간에 따라 진동하는 탐험 강도
        c_t = self._c_t()

        # UCB 기본 점수
        ucb_base = self.q_values + c_t * np.sqrt(np.log(self.t) / self.pulls)

        # 주기 보너스 (sin 텀)
        periodic = self._periodic_bonus()

        # 최종 점수
        scores = ucb_base + periodic

        max_val = np.max(scores)
        best_arms = np.where(scores == max_val)[0]
        return int(np.random.choice(best_arms))

    def getReward(self, arm: int, reward: float):
        """BaseAgent의 공통 메모리 업데이트"""
        super().getReward(arm, reward)
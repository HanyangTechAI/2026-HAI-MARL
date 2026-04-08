import numpy as np
from .base_agent import BaseAgent


class FFTPeriodicUCB(BaseAgent):
    """
    FFT-Periodic UCB 알고리즘

    Phase 1 — Warmup (round-robin):
        모든 arm을 균등하게 탐험하며 보상 히스토리 수집.
        warmup_rounds 라운드 × num_arms 스텝 동안 진행.

    Phase 2 — FFT 추정:
        각 arm의 보상 시계열에 FFT를 적용해 dominant 주기(T*)와
        amplitude, phase를 추정. sin 예측 모델 피팅.

    Phase 3 — Periodic UCB:
        score(a) = Q(a) + c * sqrt(ln(t) / N(a)) + sin_pred(a, t)
        sin_pred(a, t) = amp_a * sin(2π * t / T*_a + phi_a)

    Args:
        num_arms      : arm 개수
        c             : UCB 탐험 강도 (기본값 0.1)
        warmup_rounds : arm당 몇 라운드씩 균등 탐험할지 (기본값 30 → 30×num_arms 스텝)
        sin_weight    : sin 예측 보너스 가중치 (기본값 1.0)
        min_period    : FFT에서 고려할 최소 주기 (기본값 3)
        max_period    : FFT에서 고려할 최대 주기 (기본값 365)
        name          : 에이전트 이름
    """

    def __init__(
        self,
        num_arms: int,
        c: float = 0.1,
        warmup_rounds: int = 30,
        sin_weight: float = 1.0,
        min_period: float = 3.0,
        max_period: float = 365.0,
        name: str = "FFT_UCB",
    ):
        super().__init__(num_arms, name=name)
        self.c             = c
        self.warmup_rounds = warmup_rounds
        self.sin_weight    = sin_weight
        self.min_period    = min_period
        self.max_period    = max_period

        self.warmup_total  = warmup_rounds * num_arms  # warmup 종료 스텝

        # Phase 1: 보상 히스토리 (arm별 시계열)
        self._history: list[list[float]] = [[] for _ in range(num_arms)]

        # Phase 2: FFT 추정 결과 저장
        # 각 arm: (amplitude, period, phase) or None
        self._sin_params: list[tuple | None] = [None] * num_arms
        self._fft_done = False

        # Phase 1 round-robin 포인터
        self._rr_index = 0

    # ------------------------------------------------------------------
    # FFT 추정
    # ------------------------------------------------------------------

    def _fit_fft(self, arm: int) -> tuple[float, float, float]:
        """
        arm의 보상 시계열에 FFT를 적용해
        dominant 주기의 (amplitude, period, phase)를 반환합니다.
        """
        y = np.array(self._history[arm])
        y = y - y.mean()  # DC 성분 제거

        n = len(y)
        fft_vals = np.fft.rfft(y)
        freqs    = np.fft.rfftfreq(n)  # 주파수 (cycles/sample)

        # 주파수 → 주기 변환, 범위 필터링
        with np.errstate(divide='ignore', invalid='ignore'):
            periods = np.where(freqs > 0, 1.0 / freqs, np.inf)

        # 유효 주기 범위 마스크
        mask = (periods >= self.min_period) & (periods <= self.max_period)

        if not np.any(mask):
            # 유효한 주기 없으면 sin 보너스 비활성화
            return 0.0, 1.0, 0.0

        # dominant 주파수 (amplitude 최대)
        amplitudes = np.abs(fft_vals)
        amplitudes[~mask] = 0.0
        dominant_idx = np.argmax(amplitudes)

        amp    = amplitudes[dominant_idx] * 2 / n  # 정규화
        period = periods[dominant_idx]
        phase  = np.angle(fft_vals[dominant_idx])

        return float(amp), float(period), float(phase)

    def _run_fft_all(self):
        """모든 arm에 대해 FFT 추정 실행"""
        for arm in range(self.num_arms):
            self._sin_params[arm] = self._fit_fft(arm)
        self._fft_done = True

    # ------------------------------------------------------------------
    # sin 예측 보너스
    # ------------------------------------------------------------------

    def _sin_bonus(self, t: int) -> np.ndarray:
        """현재 스텝 t에서 각 arm의 sin 예측 보너스 반환"""
        bonuses = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            if self._sin_params[arm] is None:
                continue
            amp, period, phase = self._sin_params[arm]
            bonuses[arm] = amp * np.sin(2 * np.pi * t / period + phase)
        return bonuses * self.sin_weight

    # ------------------------------------------------------------------
    # BaseAgent 인터페이스 구현
    # ------------------------------------------------------------------

    def choice(self) -> int:
        # ── Phase 1: Warmup (round-robin) ──────────────────────────────
        if self.t < self.warmup_total:
            arm = self._rr_index % self.num_arms
            self._rr_index += 1
            return arm

        # ── FFT 추정 (warmup 직후 1회) ─────────────────────────────────
        if not self._fft_done:
            self._run_fft_all()

        # ── Phase 3: Periodic UCB ──────────────────────────────────────
        unexplored = np.where(self.pulls == 0)[0]
        if len(unexplored) > 0:
            return int(np.random.choice(unexplored))

        ucb_scores = self.q_values + self.c * np.sqrt(np.log(self.t) / self.pulls)
        sin_scores = self._sin_bonus(self.t)
        scores     = ucb_scores + sin_scores

        max_val   = np.max(scores)
        best_arms = np.where(scores == max_val)[0]
        return int(np.random.choice(best_arms))

    def getReward(self, arm: int, reward: float):
        """
        보상 업데이트 + warmup 중이면 히스토리에 기록
        """
        super().getReward(arm, reward)

        # warmup 구간에만 히스토리 수집
        if self.t <= self.warmup_total:
            self._history[arm].append(reward)
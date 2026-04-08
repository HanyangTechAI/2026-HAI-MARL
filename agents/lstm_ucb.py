import numpy as np
from collections import deque
from .base_agent import BaseAgent

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class _ArmLSTM(nn.Module):
    """
    단일 LSTM 모델 — 전체 arm의 보상 시퀀스를 입력받아
    각 arm의 다음 스텝 보상을 예측합니다.

    입력: (batch, seq_len, num_arms)
    출력: (batch, num_arms) — 각 arm의 예측 보상
    """
    def __init__(self, num_arms: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_arms,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, num_arms)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])  # 마지막 타임스텝 출력


class LSTMUCBAgent(BaseAgent):
    """
    LSTM-UCB 알고리즘

    단일 LSTM이 전체 arm의 보상 시계열을 보고 다음 스텝 보상을 예측.
    예측값(mean)과 불확실성(분산)을 결합해 UCB 점수를 계산합니다.

    선택 기준:
      score(a) = lstm_pred(a) + c * uncertainty(a)

      uncertainty(a): 최근 pred_window 스텝 동안의 예측 오차 표준편차
                      → 예측이 불안정한 arm을 더 탐험

    업데이트:
      매 스텝마다 seq_len 길이의 슬라이딩 윈도우로 온라인 학습.

    Args:
        num_arms    : arm 개수
        seq_len     : LSTM 입력 시퀀스 길이 (기본값 20)
        hidden_size : LSTM hidden 크기 (기본값 32)
        lr          : 학습률 (기본값 1e-3)
        c           : 불확실성 탐험 강도 (기본값 0.1)
        pred_window : 불확실성 계산에 사용할 최근 예측 오차 윈도우 (기본값 20)
        warmup      : LSTM 학습 시작 전 일반 UCB로 운영할 스텝 수 (기본값 seq_len+1)
        name        : 에이전트 이름
    """

    def __init__(
        self,
        num_arms: int,
        seq_len: int = 20,
        hidden_size: int = 32,
        lr: float = 1e-3,
        c: float = 0.1,
        pred_window: int = 20,
        warmup: int = None,
        name: str = "LSTM_UCB",
    ):
        super().__init__(num_arms, name=name)
        assert TORCH_AVAILABLE, "PyTorch가 설치되어 있지 않습니다. pip install torch"

        self.seq_len     = seq_len
        self.c           = c
        self.pred_window = pred_window
        self.warmup      = warmup if warmup is not None else seq_len + 1

        # 전체 arm의 보상 시계열 버퍼 (슬라이딩 윈도우)
        self._reward_buf: deque = deque(
            [np.zeros(num_arms)] * seq_len,
            maxlen=seq_len + 1  # seq + 1타겟
        )

        # 모델 & 옵티마이저
        self._model = _ArmLSTM(num_arms, hidden_size=hidden_size)
        self._opt   = optim.Adam(self._model.parameters(), lr=lr)
        self._loss_fn = nn.MSELoss()

        # 불확실성 계산용: arm별 예측 오차 히스토리
        self._pred_errors: list[deque] = [
            deque(maxlen=pred_window) for _ in range(num_arms)
        ]

        # 마지막 예측값 캐시
        self._last_pred = np.zeros(num_arms)

    # ------------------------------------------------------------------
    # LSTM 관련
    # ------------------------------------------------------------------

    def _get_sequence(self) -> torch.Tensor:
        """버퍼에서 (1, seq_len, num_arms) 텐서 반환"""
        seq = list(self._reward_buf)[:self.seq_len]
        return torch.tensor(np.array(seq), dtype=torch.float32).unsqueeze(0)

    def _predict(self) -> np.ndarray:
        """현재 시퀀스로 다음 스텝 보상 예측"""
        self._model.eval()
        with torch.no_grad():
            x = self._get_sequence()
            pred = self._model(x).squeeze(0).numpy()
        return pred

    def _update(self, actual_reward_vec: np.ndarray):
        """
        온라인 업데이트 — 버퍼의 최근 seq_len을 입력,
        실제 보상 벡터를 타겟으로 1 gradient step.
        """
        if len(self._reward_buf) < self.seq_len + 1:
            return

        self._model.train()
        x = self._get_sequence()
        y = torch.tensor(actual_reward_vec, dtype=torch.float32).unsqueeze(0)

        pred = self._model(x)
        loss = self._loss_fn(pred, y)

        self._opt.zero_grad()
        loss.backward()
        self._opt.step()

    def _uncertainty(self) -> np.ndarray:
        """arm별 예측 오차 표준편차 (불확실성 보너스)"""
        unc = np.zeros(self.num_arms)
        for i, errors in enumerate(self._pred_errors):
            if len(errors) >= 2:
                unc[i] = np.std(errors)
            else:
                unc[i] = 1.0  # 데이터 부족 → 최대 탐험
        return unc

    # ------------------------------------------------------------------
    # BaseAgent 인터페이스 구현
    # ------------------------------------------------------------------

    def choice(self) -> int:
        # warmup: 일반 UCB
        if self.t < self.warmup:
            unexplored = np.where(self.pulls == 0)[0]
            if len(unexplored) > 0:
                return int(np.random.choice(unexplored))
            ucb = self.q_values + self.c * np.sqrt(np.log(self.t + 1) / (self.pulls + 1e-9))
            return int(np.argmax(ucb))

        # LSTM 예측
        pred = self._predict()
        self._last_pred = pred

        # 불확실성 보너스
        unc = self._uncertainty()

        # 최종 점수
        scores = pred + self.c * unc

        max_val   = np.max(scores)
        best_arms = np.where(scores == max_val)[0]
        return int(np.random.choice(best_arms))

    def getReward(self, arm: int, reward: float):
        # 공통 메모리 업데이트
        super().getReward(arm, reward)

        # 실제 보상 벡터 구성 (선택된 arm만 실제값, 나머지는 예측값 유지)
        reward_vec = self._last_pred.copy()
        reward_vec[arm] = reward

        # 예측 오차 기록 (불확실성 계산용)
        pred_err = abs(self._last_pred[arm] - reward)
        self._pred_errors[arm].append(pred_err)

        # 버퍼에 추가
        self._reward_buf.append(reward_vec)

        # 온라인 업데이트
        if self.t >= self.warmup:
            self._update(reward_vec)
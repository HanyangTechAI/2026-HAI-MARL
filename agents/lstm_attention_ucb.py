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


class _SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.drop(attn_out))


class _CrossAttention(nn.Module):
    def __init__(self, d_arm: int, num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_arm, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_arm)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, S, A, D = x.shape
        x_flat = x.view(B * S, A, D)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        out = self.norm(x_flat + self.drop(attn_out))
        return out.view(B, S, A, D)


class _LSTMAttentionModel(nn.Module):
    def __init__(self, num_arms, d_model=64, d_arm=16, lstm_layers=1,
                 sa_heads=4, ca_heads=2, dropout=0.1):
        super().__init__()
        self.num_arms = num_arms
        self.d_arm    = d_arm

        self.input_proj = nn.Linear(num_arms, d_model)
        self.lstm = nn.LSTM(d_model, d_model, lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.self_attn  = _SelfAttention(d_model, sa_heads, dropout)
        self.arm_proj   = nn.Linear(d_model, num_arms * d_arm)
        self.cross_attn = _CrossAttention(d_arm, ca_heads, dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_arm, d_arm), nn.ReLU(), nn.Linear(d_arm, 1)
        )

    def forward(self, x):
        B, S, _ = x.shape
        h = self.input_proj(x)
        h, _ = self.lstm(h)
        h = self.self_attn(h)
        h_last = h[:, -1, :]
        h_arm = self.arm_proj(h_last).view(B, self.num_arms, self.d_arm)
        h_arm = self.cross_attn(h_arm.unsqueeze(1)).squeeze(1)
        return self.fc(h_arm).squeeze(-1)


class LSTMAttentionUCBAgent(BaseAgent):
    """
    LSTM + Self-Attention + Cross-Attention UCB 에이전트
    + Exponential Loss Weighting (망각 메커니즘)

    망각 메커니즘:
      학습 시 샘플에 시간 기반 감쇠 가중치를 적용합니다.
      weight(t) = gamma ^ (T - t)
      → 최근 샘플일수록 loss 기여도 높음, 오래된 샘플은 자연스럽게 망각

    Args:
        num_arms    : arm 개수
        seq_len     : 입력 시퀀스 길이 (기본값 20)
        d_model     : LSTM/Attention hidden 크기 (기본값 64)
        d_arm       : Cross-Attention arm 임베딩 크기 (기본값 16)
        lstm_layers : LSTM 레이어 수 (기본값 1)
        sa_heads    : Self-Attention 헤드 수 (기본값 4)
        ca_heads    : Cross-Attention 헤드 수 (기본값 2)
        lr          : 학습률 (기본값 1e-3)
        c           : 불확실성 탐험 강도 (기본값 0.1)
        pred_window : 불확실성 계산 윈도우 (기본값 20)
        dropout     : 드롭아웃 비율 (기본값 0.1)
        gamma       : 망각 감쇠 계수 (기본값 0.95)
                      1.0 = 망각 없음, 0.9 = 빠른 망각
        warmup      : 초기 UCB 운영 스텝 수
        name        : 에이전트 이름
    """

    def __init__(
        self,
        num_arms: int,
        seq_len: int = 20,
        d_model: int = 64,
        d_arm: int = 16,
        lstm_layers: int = 1,
        sa_heads: int = 4,
        ca_heads: int = 2,
        lr: float = 1e-3,
        c: float = 0.1,
        pred_window: int = 20,
        dropout: float = 0.1,
        gamma: float = 0.95,
        warmup: int = None,
        name: str = "LSTM_Attn_UCB",
    ):
        super().__init__(num_arms, name=name)
        assert TORCH_AVAILABLE, "PyTorch가 설치되어 있지 않습니다. pip install torch"

        self.seq_len     = seq_len
        self.c           = c
        self.pred_window = pred_window
        self.gamma       = gamma
        self.warmup      = warmup if warmup is not None else seq_len + 1

        # 보상 버퍼 — (reward_vec, timestamp) 쌍으로 저장
        self._reward_buf: deque = deque(
            [(np.zeros(num_arms), i) for i in range(seq_len)],
            maxlen=seq_len + 1
        )

        self._model = _LSTMAttentionModel(
            num_arms, d_model, d_arm, lstm_layers, sa_heads, ca_heads, dropout
        )
        self._opt     = optim.Adam(self._model.parameters(), lr=lr)
        self._loss_fn = nn.MSELoss(reduction='none')  # elementwise → 가중치 적용용

        self._pred_errors: list[deque] = [
            deque(maxlen=pred_window) for _ in range(num_arms)
        ]
        self._last_pred = np.zeros(num_arms)

    # ------------------------------------------------------------------

    def _get_sequence(self) -> tuple[torch.Tensor, torch.Tensor]:
        """버퍼에서 시퀀스와 타임스탬프 반환"""
        buf = list(self._reward_buf)[:self.seq_len]
        seq = np.array([r for r, _ in buf])
        timestamps = np.array([t for _, t in buf])
        return (
            torch.tensor(seq, dtype=torch.float32).unsqueeze(0),
            timestamps
        )

    def _predict(self) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            x, _ = self._get_sequence()
            pred = self._model(x).squeeze(0).numpy()
        return pred

    def _update(self, reward_vec: np.ndarray, current_t: int):
        """
        Exponential Loss Weighting 적용 업데이트
        weight = gamma ^ (current_t - timestamp)
        """
        if len(self._reward_buf) < self.seq_len + 1:
            return

        self._model.train()
        x, timestamps = self._get_sequence()
        y = torch.tensor(reward_vec, dtype=torch.float32).unsqueeze(0)

        pred = self._model(x)

        # elementwise MSE → (1, num_arms)
        loss_elem = self._loss_fn(pred, y)

        # 시퀀스의 마지막 타임스탬프 기준 감쇠 가중치
        # 현재 스텝에 가까울수록 weight=1, 멀수록 gamma^k
        age = current_t - timestamps[-1]  # 마지막 입력 스텝 기준
        weight = self.gamma ** max(age, 0)
        weighted_loss = (loss_elem * weight).mean()

        self._opt.zero_grad()
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._opt.step()

    def _uncertainty(self) -> np.ndarray:
        unc = np.zeros(self.num_arms)
        for i, errors in enumerate(self._pred_errors):
            unc[i] = np.std(errors) if len(errors) >= 2 else 1.0
        return unc

    # ------------------------------------------------------------------

    def choice(self) -> int:
        if self.t < self.warmup:
            unexplored = np.where(self.pulls == 0)[0]
            if len(unexplored) > 0:
                return int(np.random.choice(unexplored))
            ucb = self.q_values + self.c * np.sqrt(np.log(self.t + 1) / (self.pulls + 1e-9))
            return int(np.argmax(ucb))

        pred = self._predict()
        self._last_pred = pred
        scores = pred + self.c * self._uncertainty()

        best_arms = np.where(scores == np.max(scores))[0]
        return int(np.random.choice(best_arms))

    def getReward(self, arm: int, reward: float):
        super().getReward(arm, reward)

        reward_vec = self._last_pred.copy()
        reward_vec[arm] = reward

        self._pred_errors[arm].append(abs(self._last_pred[arm] - reward))
        self._reward_buf.append((reward_vec, self.t))  # timestamp 함께 저장

        if self.t >= self.warmup:
            self._update(reward_vec, self.t)
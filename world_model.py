"""
WorldModelAgent for HAI-MARL (Multi-Armed Bandit Environment)
=============================================================
환경 분석 요약:
  - MAB 환경: 8개 arm (Stationary×2, EventShock×2, Trend×2, Switch×2)
  - Action space : Discrete(8)  → arm index 선택
  - Observation  : scalar reward (float) + 내부 통계 누적
  - Collision    : reward / (n_collisions ^ 1.5)  슬리피지 패널티
  - Horizon      : 1941 steps (Walmart 데이터 기반)
  - 기존 인터페이스: agent.choice() → int,  agent.getReward(arm, reward)

State 벡터 구성 (풍부한 state):
  [히스토리 파트] 최근 H스텝의 (one-hot action ‖ reward)  → H × (K+1)
  [통계 파트]     각 arm별 (평균보상, 선택횟수_log, 분산)   → K × 3
  전체 obs_dim = H*(K+1) + K*3

파라미터:
  K  = 8   (num_arms)
  H  = 16  (history window)
  obs_dim = 16*(8+1) + 8*3 = 144 + 24 = 168
  z_dim   = 32  (V 인코더 잠재 차원)
  h_dim   = 64  (M MDN-RNN hidden 차원)
  n_mix   = 5   (MDN 가우시안 믹스처 수)
  action_dim = 8

사용법:
  agent = WorldModelAgent(num_arms=8)
  arm   = agent.choice()         # → int (0~7)
  agent.getReward(arm, reward)   # 환경으로부터 보상 수신

학습 루프는 파일 하단 train_world_model() 함수 참조.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque


# ─────────────────────────────────────────────
# 하이퍼파라미터 (한 곳에서 관리)
# ─────────────────────────────────────────────
class Config:
    # 환경
    K          = 8      # num_arms (MAB arm 수)
    HORIZON    = 1941   # Walmart 시뮬레이션 길이
    N_AGENTS   = 9      # CustomEvaluator에 투입할 에이전트 수 (참고용)

    # State 구성
    H          = 16     # 히스토리 윈도우 크기
    OBS_DIM    = H * (K + 1) + K * 3  # = 168

    # V 모듈 (MLP Encoder)
    Z_DIM      = 32     # 잠재 벡터 차원
    V_HIDDEN   = 128    # V 인코더/디코더 히든 크기

    # M 모듈 (MDN-RNN)
    H_DIM      = 64     # LSTM hidden 차원
    N_MIX      = 5      # MDN 가우시안 믹스처 수
    M_HIDDEN   = 128    # MDN 헤드 히든 크기

    # C 모듈 (Controller)
    # 입력: z_dim + h_dim = 32 + 64 = 96
    C_INPUT    = Z_DIM + H_DIM  # = 96

    # 학습
    LR_V       = 3e-4
    LR_M       = 3e-4
    LR_C       = 1e-3
    BATCH_SIZE = 64
    SEQ_LEN    = 32     # M 학습용 시퀀스 길이
    GAMMA      = 0.99   # 보상 할인율
    REPLAY_MIN = 256    # 최소 replay buffer 크기

    # 탐험
    EPSILON_START = 1.0
    EPSILON_END   = 0.05
    EPSILON_DECAY = 0.995


cfg = Config()


# ─────────────────────────────────────────────
# V 모듈: MLP Encoder / Decoder
# ─────────────────────────────────────────────
class VEncoder(nn.Module):
    """
    obs_dim(168) → z_dim(32) 압축 인코더.
    VAE 스타일: mean과 logvar를 출력해 reparameterization trick 사용.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.OBS_DIM, cfg.V_HIDDEN),
            nn.LayerNorm(cfg.V_HIDDEN),
            nn.ReLU(),
            nn.Linear(cfg.V_HIDDEN, cfg.V_HIDDEN // 2),
            nn.ReLU(),
        )
        self.fc_mean   = nn.Linear(cfg.V_HIDDEN // 2, cfg.Z_DIM)
        self.fc_logvar = nn.Linear(cfg.V_HIDDEN // 2, cfg.Z_DIM)

    def forward(self, obs):
        """
        obs : (batch, obs_dim=168)
        returns z_mean, z_logvar, z (reparameterized)
        """
        h = self.net(obs)
        z_mean   = self.fc_mean(h)
        z_logvar = self.fc_logvar(h).clamp(-4, 4)  # 수치 안정성
        # Reparameterization
        if self.training:
            eps = torch.randn_like(z_mean)
            z = z_mean + eps * (0.5 * z_logvar).exp()
        else:
            z = z_mean
        return z_mean, z_logvar, z


class VDecoder(nn.Module):
    """
    z_dim(32) → obs_dim(168) 복원 디코더.
    사전학습(reconstruction loss) 및 world model 검증용.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.Z_DIM, cfg.V_HIDDEN // 2),
            nn.ReLU(),
            nn.Linear(cfg.V_HIDDEN // 2, cfg.V_HIDDEN),
            nn.ReLU(),
            nn.Linear(cfg.V_HIDDEN, cfg.OBS_DIM),
        )

    def forward(self, z):
        return self.net(z)


# ─────────────────────────────────────────────
# M 모듈: MDN-RNN
# ─────────────────────────────────────────────
class MDNRNN(nn.Module):
    """
    MDN-RNN: 과거 (z, action) 시퀀스로 미래 z 분포 예측.

    입력: z_t (z_dim=32) + one-hot action (K=8) = 40차원
    LSTM hidden: h_dim=64
    출력: N_MIX 개의 (pi, mu, sigma) — 다음 z의 Gaussian 믹스처
    """
    def __init__(self):
        super().__init__()
        self.input_dim = cfg.Z_DIM + cfg.K  # 32 + 8 = 40

        self.lstm = nn.LSTM(
            input_size  = self.input_dim,
            hidden_size = cfg.H_DIM,
            num_layers  = 1,
            batch_first = True,
        )

        # MDN 헤드: hidden → (pi, mu, sigma) × n_mix × z_dim
        mdn_out = cfg.N_MIX * (1 + cfg.Z_DIM + cfg.Z_DIM)  # pi + mu + sigma
        self.mdn_head = nn.Sequential(
            nn.Linear(cfg.H_DIM, cfg.M_HIDDEN),
            nn.ReLU(),
            nn.Linear(cfg.M_HIDDEN, mdn_out),
        )

    def forward(self, z_seq, action_seq, hidden=None):
        """
        z_seq      : (batch, seq_len, z_dim=32)
        action_seq : (batch, seq_len) — arm indices (0~7)
        hidden     : LSTM hidden state (None이면 0으로 초기화)

        returns:
          pi    : (batch, seq_len, n_mix)        — 믹스처 가중치
          mu    : (batch, seq_len, n_mix, z_dim) — 각 가우시안 평균
          sigma : (batch, seq_len, n_mix, z_dim) — 각 가우시안 표준편차
          hidden: 업데이트된 LSTM hidden state
        """
        B, T, _ = z_seq.shape

        # action one-hot 인코딩
        a_onehot = F.one_hot(action_seq.long(), cfg.K).float()  # (B, T, K)

        # LSTM 입력 조합
        lstm_in = torch.cat([z_seq, a_onehot], dim=-1)  # (B, T, 40)

        lstm_out, hidden = self.lstm(lstm_in, hidden)    # (B, T, h_dim)

        # MDN 파라미터 추출
        mdn_params = self.mdn_head(lstm_out)             # (B, T, n_mix*(1+z+z))

        # 분리
        n = cfg.N_MIX
        z = cfg.Z_DIM
        pi_raw = mdn_params[..., :n]                     # (B, T, n_mix)
        mu     = mdn_params[..., n : n + n*z].view(B, T, n, z)
        sigma  = mdn_params[..., n + n*z :].view(B, T, n, z)

        pi    = F.softmax(pi_raw, dim=-1)
        sigma = F.softplus(sigma) + 1e-6  # 양수 보장

        return pi, mu, sigma, hidden

    def get_hidden_single(self, z, action, hidden=None):
        """
        단일 스텝 추론용 (choice() 시 사용).
        z      : (z_dim,)  — 1D 텐서
        action : int
        returns h_vec (h_dim,), hidden
        """
        z_in = z.unsqueeze(0).unsqueeze(0)           # (1, 1, z_dim)
        a_in = torch.tensor([[action]])               # (1, 1)
        _, _, _, hidden = self.forward(z_in, a_in, hidden)
        # hidden[0] : (1, 1, h_dim) → (h_dim,)
        h_vec = hidden[0].squeeze(0).squeeze(0)
        return h_vec, hidden


# ─────────────────────────────────────────────
# C 모듈: Controller
# ─────────────────────────────────────────────
class Controller(nn.Module):
    """
    [z_t ‖ h_t] → action logits (K=8개 arm 확률).
    입력 차원: z_dim + h_dim = 32 + 64 = 96
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.C_INPUT, 64),   # 96 → 64
            nn.ReLU(),
            nn.Linear(64, cfg.K),          # 64 → 8
        )

    def forward(self, z, h):
        """
        z : (batch, z_dim=32)  또는 (z_dim,)
        h : (batch, h_dim=64)  또는 (h_dim,)
        returns logits : (batch, K=8)
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
            h = h.unsqueeze(0)
        x = torch.cat([z, h], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────
class ReplayBuffer:
    """
    (obs, action, reward, next_obs) 저장.
    M 학습을 위해 시퀀스(deque 순서) 유지.
    """
    def __init__(self, maxlen=10000):
        self.buffer = deque(maxlen=maxlen)

    def push(self, obs, action, reward, next_obs):
        self.buffer.append((obs, action, reward, next_obs))

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        obs, actions, rewards, next_obs = zip(*batch)
        return (
            torch.FloatTensor(np.array(obs)),
            torch.LongTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.FloatTensor(np.array(next_obs)),
        )

    def sample_sequences(self, batch_size, seq_len):
        """M 학습용: (obs, action) 시퀀스 샘플링"""
        max_start = len(self.buffer) - seq_len
        if max_start <= 0:
            return None
        starts = np.random.choice(max_start, batch_size, replace=True)
        obs_seqs, action_seqs, reward_seqs = [], [], []
        for s in starts:
            seq = [self.buffer[s + i] for i in range(seq_len)]
            obs_seqs.append([t[0] for t in seq])
            action_seqs.append([t[1] for t in seq])
            reward_seqs.append([t[2] for t in seq])
        return (
            torch.FloatTensor(np.array(obs_seqs)),    # (B, T, obs_dim)
            torch.LongTensor(np.array(action_seqs)),  # (B, T)
            torch.FloatTensor(np.array(reward_seqs)), # (B, T)
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
# MDN Loss
# ─────────────────────────────────────────────
def mdn_loss(pi, mu, sigma, target):
    """
    Negative log-likelihood of Gaussian mixture.
    pi     : (B, T, n_mix)
    mu     : (B, T, n_mix, z_dim)
    sigma  : (B, T, n_mix, z_dim)
    target : (B, T, z_dim)

    returns scalar loss
    """
    target_expanded = target.unsqueeze(2)  # (B, T, 1, z_dim)
    # log N(target | mu, sigma) per component per dim
    log_prob = -0.5 * ((target_expanded - mu) / sigma) ** 2 \
               - sigma.log() - 0.5 * np.log(2 * np.pi)
    # sum over z_dim, add log pi
    log_prob = log_prob.sum(-1) + pi.log()          # (B, T, n_mix)
    # log-sum-exp over mixtures
    nll = -torch.logsumexp(log_prob, dim=-1)         # (B, T)
    return nll.mean()


# ─────────────────────────────────────────────
# WorldModelAgent (기존 인터페이스 호환)
# ─────────────────────────────────────────────
class WorldModelAgent:
    """
    Ha & Schmidhuber (2018) World Model을 MAB 환경에 적용한 에이전트.

    기존 CustomEvaluator 인터페이스 완전 호환:
      arm  = agent.choice()           # arm 선택 (0 ~ K-1)
      agent.getReward(arm, reward)    # 보상 수신 및 내부 업데이트
    """

    def __init__(self, num_arms=cfg.K, name="WorldModelAgent", device="cpu"):
        self.name      = name
        self.K         = num_arms
        self.device    = torch.device(device)
        self.t         = 0          # 현재 스텝
        self.epsilon   = cfg.EPSILON_START

        # ── 모듈 초기화 ──────────────────────────
        self.V_enc = VEncoder().to(self.device)
        self.V_dec = VDecoder().to(self.device)
        self.M     = MDNRNN().to(self.device)
        self.C     = Controller().to(self.device)

        # ── 옵티마이저 ───────────────────────────
        self.opt_V = Adam(
            list(self.V_enc.parameters()) + list(self.V_dec.parameters()),
            lr=cfg.LR_V
        )
        self.opt_M = Adam(self.M.parameters(), lr=cfg.LR_M)
        self.opt_C = Adam(self.C.parameters(), lr=cfg.LR_C)

        # ── 내부 상태 ────────────────────────────
        # 히스토리 윈도우: deque of (arm_idx, reward)
        self._history  = deque(maxlen=cfg.H)
        # 채우기: (arm=0, reward=0) × H
        for _ in range(cfg.H):
            self._history.append((0, 0.0))

        # arm별 누적 통계
        self._counts   = np.zeros(self.K)           # 선택 횟수
        self._sum_r    = np.zeros(self.K)           # 보상 합
        self._sum_r2   = np.zeros(self.K)           # 보상 제곱합 (분산용)

        # LSTM hidden state
        self._lstm_hidden = None
        self._last_z      = torch.zeros(cfg.Z_DIM).to(self.device)
        self._last_h      = torch.zeros(cfg.H_DIM).to(self.device)
        self._last_action = 0

        # Replay buffer
        self.replay = ReplayBuffer(maxlen=20000)
        self._last_obs = None

        # 학습 손실 기록
        self.loss_v_log = []
        self.loss_m_log = []
        self.loss_c_log = []

    # ─── State 벡터 빌드 ───────────────────────
    def _build_obs(self):
        """
        현재 내부 통계로 obs 벡터(168차원) 구성.

        [히스토리 파트] H × (K+1):
          각 스텝: one-hot(action, K) + scalar_reward  → K+1 차원
        [통계 파트] K × 3:
          arm별 (평균보상, log(1+count), 분산)
        """
        # 히스토리
        hist_parts = []
        for (arm, r) in self._history:
            onehot = np.zeros(self.K, dtype=np.float32)
            onehot[arm] = 1.0
            hist_parts.append(np.append(onehot, r))
        hist_vec = np.concatenate(hist_parts)  # H*(K+1)

        # 통계
        avg_r  = np.where(self._counts > 0, self._sum_r / self._counts, 0.0)
        log_n  = np.log1p(self._counts)
        var_r  = np.where(
            self._counts > 1,
            self._sum_r2 / self._counts - avg_r**2,
            0.0
        )
        stat_vec = np.stack([avg_r, log_n, var_r], axis=1).flatten()  # K*3

        return np.concatenate([hist_vec, stat_vec]).astype(np.float32)

    # ─── 기존 인터페이스: choice() ─────────────
    def choice(self):
        """
        현재 관측을 인코딩하고 Controller로 arm 선택.
        ε-greedy 탐험 적용.
        """
        self.V_enc.eval()
        self.C.eval()

        obs = self._build_obs()
        self._last_obs = obs.copy()

        obs_t = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            _, _, z = self.V_enc(obs_t.unsqueeze(0))
            z = z.squeeze(0)
            logits = self.C(z, self._last_h)
            probs  = F.softmax(logits, dim=-1).squeeze(0)

        # ε-greedy
        if np.random.random() < self.epsilon:
            arm = np.random.randint(self.K)
        else:
            arm = probs.argmax().item()

        self._last_z      = z
        self._last_action = arm
        return arm

    # ─── 기존 인터페이스: getReward() ──────────
    def getReward(self, arm, reward):
        """
        보상 수신 → 내부 통계 업데이트 → 모듈 학습.
        """
        # 통계 업데이트
        self._counts[arm]  += 1
        self._sum_r[arm]   += reward
        self._sum_r2[arm]  += reward ** 2
        self._history.append((arm, float(reward)))

        # M 모듈: LSTM hidden 업데이트
        self.M.eval()
        with torch.no_grad():
            self._last_h, self._lstm_hidden = self.M.get_hidden_single(
                self._last_z, arm, self._lstm_hidden
            )

        # 다음 obs 빌드 후 replay 저장
        next_obs = self._build_obs()
        if self._last_obs is not None:
            self.replay.push(
                self._last_obs,
                arm,
                float(reward),
                next_obs
            )

        # ε decay
        self.epsilon = max(
            cfg.EPSILON_END,
            self.epsilon * cfg.EPSILON_DECAY
        )

        # 학습 트리거 (충분한 데이터 쌓인 이후)
        if len(self.replay) >= cfg.REPLAY_MIN:
            self._train_step()

        self.t += 1

    # ─── 학습 스텝 ─────────────────────────────
    def _train_step(self):
        """V, M, C 모듈을 각각 1 gradient step 업데이트."""
        self._train_V()
        self._train_M()
        self._train_C()

    def _train_V(self):
        """VAE reconstruction loss + KL divergence."""
        self.V_enc.train()
        self.V_dec.train()
        self.opt_V.zero_grad()

        obs, _, _, next_obs = self.replay.sample(cfg.BATCH_SIZE)
        obs = obs.to(self.device)

        z_mean, z_logvar, z = self.V_enc(obs)
        obs_recon = self.V_dec(z)

        recon_loss = F.mse_loss(obs_recon, obs)
        kl_loss    = -0.5 * (1 + z_logvar - z_mean**2 - z_logvar.exp()).mean()
        loss       = recon_loss + 0.001 * kl_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.V_enc.parameters()) + list(self.V_dec.parameters()), 1.0
        )
        self.opt_V.step()
        self.loss_v_log.append(loss.item())

    def _train_M(self):
        """MDN-RNN: 다음 z 예측의 NLL 최소화."""
        result = self.replay.sample_sequences(cfg.BATCH_SIZE, cfg.SEQ_LEN)
        if result is None:
            return
        obs_seq, action_seq, _ = result
        obs_seq    = obs_seq.to(self.device)
        action_seq = action_seq.to(self.device)

        self.V_enc.eval()
        self.M.train()
        self.opt_M.zero_grad()

        with torch.no_grad():
            B, T, D = obs_seq.shape
            obs_flat = obs_seq.view(B * T, D)
            _, _, z_flat = self.V_enc(obs_flat)
            z_seq = z_flat.view(B, T, cfg.Z_DIM)

        # 입력: t=0..T-2, 타깃: t=1..T-1
        z_in     = z_seq[:, :-1, :]       # (B, T-1, z_dim)
        z_target = z_seq[:, 1:, :]        # (B, T-1, z_dim)
        a_in     = action_seq[:, :-1]     # (B, T-1)

        pi, mu, sigma, _ = self.M(z_in, a_in)
        loss = mdn_loss(pi, mu, sigma, z_target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.M.parameters(), 1.0)
        self.opt_M.step()
        self.loss_m_log.append(loss.item())

    def _train_C(self):
        """
        Controller: DQN 스타일 Q-learning.
        Q(s,a) = r + γ * max Q(s',a')
        """
        self.V_enc.eval()
        self.C.train()
        self.opt_C.zero_grad()

        obs, actions, rewards, next_obs = self.replay.sample(cfg.BATCH_SIZE)
        obs      = obs.to(self.device)
        next_obs = next_obs.to(self.device)
        rewards  = rewards.to(self.device)
        actions  = actions.to(self.device)

        with torch.no_grad():
            _, _, z      = self.V_enc(obs)
            _, _, z_next = self.V_enc(next_obs)
            # h는 현재 간단히 0벡터 사용 (완전 학습 시 M과 결합)
            h      = torch.zeros(obs.size(0), cfg.H_DIM).to(self.device)
            h_next = torch.zeros(obs.size(0), cfg.H_DIM).to(self.device)

        # 현재 Q값
        q_all    = self.C(z, h)                            # (B, K)
        q_taken  = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 타깃 Q값
        with torch.no_grad():
            q_next = self.C(z_next, h_next).max(dim=1).values
            q_target = rewards + cfg.GAMMA * q_next

        loss = F.smooth_l1_loss(q_taken, q_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.C.parameters(), 1.0)
        self.opt_C.step()
        self.loss_c_log.append(loss.item())

    # ─── 모델 저장/불러오기 ─────────────────────
    def save(self, path="world_model_agent.pt"):
        torch.save({
            "V_enc": self.V_enc.state_dict(),
            "V_dec": self.V_dec.state_dict(),
            "M":     self.M.state_dict(),
            "C":     self.C.state_dict(),
        }, path)
        print(f"[WorldModelAgent] 모델 저장 완료 → {path}")

    def load(self, path="world_model_agent.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.V_enc.load_state_dict(ckpt["V_enc"])
        self.V_dec.load_state_dict(ckpt["V_dec"])
        self.M.load_state_dict(ckpt["M"])
        self.C.load_state_dict(ckpt["C"])
        print(f"[WorldModelAgent] 모델 로드 완료 ← {path}")


# ─────────────────────────────────────────────
# main_time.py에서 사용하는 방법 (드롭인 교체)
# ─────────────────────────────────────────────
def create_world_model_agents(num_arms=cfg.K, n_agents=1, device="cpu"):
    """
    main_time.py의 agents 리스트에 WorldModelAgent 추가 예시:

        from agents.world_model_agent import create_world_model_agents
        wm_agents = create_world_model_agents(num_arms=env.nbArms, n_agents=2)
        agents = [...기존 에이전트..., *wm_agents]
    """
    return [
        WorldModelAgent(
            num_arms=num_arms,
            name=f"WorldModel_{i}",
            device=device
        )
        for i in range(n_agents)
    ]


# ─────────────────────────────────────────────
# 독립 학습 루프 (사전학습용)
# ─────────────────────────────────────────────
def pretrain_V(agent, n_steps=500):
    """
    랜덤 정책으로 rollout을 수집하며 V 모듈만 사전학습.
    실제 환경 없이 랜덤 obs로 reconstruction 검증.
    """
    print("[Pretrain V] 랜덤 obs로 V 인코더 워밍업 중...")
    agent.V_enc.train()
    agent.V_dec.train()
    for step in range(n_steps):
        # 랜덤 obs 생성 (실제 환경 rollout으로 교체 가능)
        fake_obs = torch.FloatTensor(
            np.random.randn(cfg.BATCH_SIZE, cfg.OBS_DIM).astype(np.float32)
        ).to(agent.device)

        agent.opt_V.zero_grad()
        z_mean, z_logvar, z = agent.V_enc(fake_obs)
        recon = agent.V_dec(z)
        loss  = F.mse_loss(recon, fake_obs)
        loss.backward()
        agent.opt_V.step()

        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}/{n_steps} | V loss: {loss.item():.6f}")
    print("[Pretrain V] 완료!\n")


# ─────────────────────────────────────────────
# 파라미터 정보 출력
# ─────────────────────────────────────────────
def print_model_info():
    agent = WorldModelAgent()
    modules = {
        "V Encoder": agent.V_enc,
        "V Decoder": agent.V_dec,
        "M MDN-RNN": agent.M,
        "C Controller": agent.C,
    }
    print("=" * 55)
    print("  WorldModelAgent 파라미터 요약")
    print("=" * 55)
    print(f"  obs_dim  = H×(K+1) + K×3 = {cfg.H}×{cfg.K+1} + {cfg.K}×3 = {cfg.OBS_DIM}")
    print(f"  z_dim    = {cfg.Z_DIM}")
    print(f"  h_dim    = {cfg.H_DIM}")
    print(f"  n_mix    = {cfg.N_MIX}")
    print(f"  C_input  = z_dim + h_dim = {cfg.C_INPUT}")
    print("-" * 55)
    total = 0
    for name, module in modules.items():
        n = sum(p.numel() for p in module.parameters())
        total += n
        print(f"  {name:<18} : {n:>8,} params")
    print("-" * 55)
    print(f"  {'합계':<18} : {total:>8,} params")
    print("=" * 55)


if __name__ == "__main__":
    print_model_info()
    print()

    # 단일 에이전트 동작 테스트
    agent = WorldModelAgent(num_arms=8, name="WM_Test")
    print(f"[테스트] obs_dim={cfg.OBS_DIM}, choice 실행...")
    for step in range(5):
        arm    = agent.choice()
        reward = np.random.normal(0.1, 0.05)   # 가짜 보상
        agent.getReward(arm, reward)
        print(f"  Step {step+1}: arm={arm}, reward={reward:.5f}, ε={agent.epsilon:.3f}")

    print("\n[완료] WorldModelAgent 정상 동작 확인!")
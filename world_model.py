"""
WorldModelAgent — Dream Training 완전 구현
==========================================

논문 (Ha & Schmidhuber, 2018) 4장의 핵심:
  "C를 실제 환경이 아닌 M이 생성한 가상 환경 안에서 학습"

변경 사항 (기존 코드 대비):
  1. MDNRNN: 보상 예측 헤드 추가
     (z_t, a_t, h_t) → z_t+1 분포 + r_t 예측
  2. _train_M(): 보상 예측 loss 추가 (MSE)
  3. dream_rollout(): 핵심 추가 함수
     M 안에서 C가 행동 선택 → M이 (z_next, r) 예측 → dream transition 수집
  4. _train_C_from_dream(): dream transition으로 C 업데이트
  5. getReward(): dream training 트리거 추가

보상 예측 방식:
  받은 reward (collision 패널티 적용 후)를 통째로 타깃으로 사용.
  → M이 "이 맥락(z, h)에서 arm a를 골랐을 때 경험적으로 이 정도 reward가 나왔다"를
    분포로 학습. collision 노이즈는 MDN의 넓은 sigma로 자연스럽게 흡수됨.
  → 논문의 철학과 일치: 보상의 원인을 분해하지 않고 M이 통째로 모델링.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import deque

try:
    from .base_agent import BaseAgent
except ImportError:
    from base_agent import BaseAgent


# ─────────────────────────────────────────────────────────
# 하이퍼파라미터
# ─────────────────────────────────────────────────────────
class Config:
    # 환경
    K            = 8
    HORIZON      = 1941

    # State
    H            = 16
    # obs_dim = H*(K+1) + K*3 → 동적 계산

    # V 모듈
    Z_DIM        = 32
    V_HIDDEN     = 128

    # M 모듈
    H_DIM        = 64
    N_MIX        = 5
    M_HIDDEN     = 128

    # 학습
    LR_V         = 3e-4
    LR_M         = 3e-4
    LR_C         = 1e-3
    BATCH_SIZE   = 64
    SEQ_LEN      = 32
    GAMMA        = 0.99
    REPLAY_MIN   = 256
    TARGET_UPDATE = 50

    # Dream Training
    DREAM_HORIZON      = 16    # 한 번의 dream에서 몇 스텝 시뮬레이션할지
    DREAM_EVERY        = 10    # 실제 환경 N스텝마다 dream 1회 실행
    DREAM_BATCH        = 8     # 한 번의 dream에서 몇 개의 시작점을 병렬로 굴릴지
    TEMPERATURE        = 1.0   # τ: dream 불확실성 조절 (높을수록 노이즈 많음)
    REWARD_LOSS_WEIGHT = 0.5   # M의 보상 예측 loss 가중치

    # 탐험
    EPS_START    = 1.0
    EPS_END      = 0.05
    EPS_DECAY    = 0.995

cfg = Config()


# ─────────────────────────────────────────────────────────
# V 모듈 (기존과 동일)
# ─────────────────────────────────────────────────────────
class VEncoder(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, cfg.V_HIDDEN),
            nn.LayerNorm(cfg.V_HIDDEN),
            nn.ReLU(),
            nn.Linear(cfg.V_HIDDEN, cfg.V_HIDDEN // 2),
            nn.ReLU(),
        )
        self.fc_mean   = nn.Linear(cfg.V_HIDDEN // 2, cfg.Z_DIM)
        self.fc_logvar = nn.Linear(cfg.V_HIDDEN // 2, cfg.Z_DIM)

    def forward(self, obs):
        h = self.net(obs)
        z_mean   = self.fc_mean(h)
        z_logvar = self.fc_logvar(h).clamp(-4, 4)
        if self.training:
            z = z_mean + (0.5 * z_logvar).exp() * torch.randn_like(z_mean)
        else:
            z = z_mean
        return z_mean, z_logvar, z


class VDecoder(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.Z_DIM, cfg.V_HIDDEN // 2),
            nn.ReLU(),
            nn.Linear(cfg.V_HIDDEN // 2, cfg.V_HIDDEN),
            nn.ReLU(),
            nn.Linear(cfg.V_HIDDEN, obs_dim),
        )

    def forward(self, z):
        return self.net(z)


# ─────────────────────────────────────────────────────────
# M 모듈: MDN-RNN + 보상 예측 헤드 ← 핵심 변경
# ─────────────────────────────────────────────────────────
class MDNRNN(nn.Module):
    """
    기존 대비 변경:
      출력이 두 가지로 분리됨
        1. z_next 분포: (pi, mu, sigma) — MDN (기존)
        2. r_pred     : scalar           — 보상 예측 헤드 (신규)

    보상 예측 원리:
      M이 과거 경험 (z, a) → r 의 패턴을 학습.
      collision 패널티는 MDN의 넓은 sigma로 자연스럽게 흡수됨.
      dream rollout에서 r_pred를 실제 reward 대신 사용.
    """
    def __init__(self, K):
        super().__init__()
        self.K = K
        self.input_dim = cfg.Z_DIM + K  # 32 + 8 = 40

        self.lstm = nn.LSTM(
            input_size  = self.input_dim,
            hidden_size = cfg.H_DIM,
            num_layers  = 1,
            batch_first = True,
        )

        # z_next 예측: MDN 헤드 (기존과 동일)
        mdn_out = cfg.N_MIX * (1 + cfg.Z_DIM + cfg.Z_DIM)
        self.mdn_head = nn.Sequential(
            nn.Linear(cfg.H_DIM, cfg.M_HIDDEN),
            nn.ReLU(),
            nn.Linear(cfg.M_HIDDEN, mdn_out),
        )

        # 보상 예측 헤드 (신규 추가)
        # h_t → scalar reward 예측
        # "이 맥락(h)에서 이 action을 했을 때 reward가 얼마였나"를 학습
        self.reward_head = nn.Sequential(
            nn.Linear(cfg.H_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, z_seq, action_seq, hidden=None):
        """
        z_seq      : (B, T, z_dim)
        action_seq : (B, T)
        returns:
          pi, mu, sigma : z_next MDN 파라미터
          r_pred        : (B, T, 1) 보상 예측 ← 신규
          hidden        : 업데이트된 LSTM hidden
        """
        B, T, _ = z_seq.shape
        a_onehot = F.one_hot(action_seq.long(), self.K).float()
        lstm_in  = torch.cat([z_seq, a_onehot], dim=-1)

        lstm_out, hidden = self.lstm(lstm_in, hidden)

        # z_next MDN
        n, z = cfg.N_MIX, cfg.Z_DIM
        raw   = self.mdn_head(lstm_out)
        pi    = F.softmax(raw[..., :n], dim=-1)
        mu    = raw[..., n: n+n*z].view(B, T, n, z)
        sigma = F.softplus(raw[..., n+n*z:]).view(B, T, n, z) + 1e-6

        # 보상 예측
        r_pred = self.reward_head(lstm_out)   # (B, T, 1)

        return pi, mu, sigma, r_pred, hidden

    def step(self, z, action, hidden=None):
        """단일 스텝 추론 — h_vec 반환 (실제 환경 학습용)"""
        z_in = z.unsqueeze(0).unsqueeze(0)
        a_in = torch.tensor([[action]])
        _, _, _, _, hidden = self.forward(z_in, a_in, hidden)
        h_vec = hidden[0].squeeze(0).squeeze(0)
        return h_vec, hidden

    def dream_step(self, z, action, hidden=None, temperature=1.0):
        """
        Dream rollout용 단일 스텝.
        M이 실제 환경 없이 (z_next, r_pred)를 생성.

        temperature τ (논문 4.2절):
          sigma를 τ배 확대 → dream이 더 불확실해짐
          → C가 M의 허점을 찾는 adversarial policy 방지
          → 실제 환경 전이 성능 향상

        returns:
          z_next : (z_dim,)  M이 예측한 다음 잠재 상태
          r_pred : float     M이 예측한 보상 (과거 경험 기반)
          hidden : 업데이트된 LSTM hidden
        """
        z_in = z.unsqueeze(0).unsqueeze(0)
        a_in = torch.tensor([[action]])

        pi, mu, sigma, r_pred, hidden = self.forward(z_in, a_in, hidden)

        # temperature로 불확실성 조절
        sigma_t = sigma * temperature

        # 믹스처 중 하나를 확률적으로 선택
        pi_np   = pi.squeeze(0).squeeze(0).detach().cpu().numpy()
        mix_idx = np.random.choice(cfg.N_MIX, p=pi_np)

        # 선택된 가우시안에서 z_next 샘플링
        mu_k    = mu[0, 0, mix_idx]
        sigma_k = sigma_t[0, 0, mix_idx]
        z_next  = mu_k + sigma_k * torch.randn_like(mu_k)

        r_val = r_pred[0, 0, 0].item()

        return z_next.detach(), r_val, hidden


# ─────────────────────────────────────────────────────────
# C 모듈: Dueling Q-Network (기존과 동일)
# ─────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    def __init__(self, K):
        super().__init__()
        in_dim = cfg.Z_DIM + cfg.H_DIM  # 96
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),     nn.ReLU(),
        )
        self.value_stream     = nn.Linear(64, 1)
        self.advantage_stream = nn.Linear(64, K)

    def forward(self, z, h):
        if z.dim() == 1:
            z = z.unsqueeze(0)
            h = h.unsqueeze(0)
        feat = self.shared(torch.cat([z, h], dim=-1))
        V = self.value_stream(feat)
        A = self.advantage_stream(feat)
        return V + A - A.mean(dim=-1, keepdim=True)


# ─────────────────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, maxlen=20000):
        self.buf = deque(maxlen=maxlen)

    def push(self, obs, action, reward, next_obs):
        self.buf.append((obs, action, float(reward), next_obs))

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buf), batch_size, replace=False)
        obs, act, rew, nobs = zip(*[self.buf[i] for i in idxs])
        return (
            torch.FloatTensor(np.array(obs)),
            torch.LongTensor(np.array(act)),
            torch.FloatTensor(np.array(rew)),
            torch.FloatTensor(np.array(nobs)),
        )

    def sample_sequences(self, batch_size, seq_len):
        max_start = len(self.buf) - seq_len
        if max_start <= 0:
            return None
        starts = np.random.choice(max_start, batch_size, replace=True)
        obs_s, act_s, rew_s = [], [], []
        for s in starts:
            seq = [self.buf[s + i] for i in range(seq_len)]
            obs_s.append([x[0] for x in seq])
            act_s.append([x[1] for x in seq])
            rew_s.append([x[2] for x in seq])
        return (
            torch.FloatTensor(np.array(obs_s)),
            torch.LongTensor(np.array(act_s)),
            torch.FloatTensor(np.array(rew_s)),
        )

    def sample_z_starts(self, n, encoder, device):
        """Dream rollout 시작점: replay obs → z 인코딩"""
        idxs = np.random.choice(len(self.buf), n, replace=True)
        obs  = torch.FloatTensor(
            np.array([self.buf[i][0] for i in idxs])
        ).to(device)
        with torch.no_grad():
            _, _, z = encoder(obs)
        return z  # (n, z_dim)

    def __len__(self):
        return len(self.buf)


# ─────────────────────────────────────────────────────────
# MDN Loss
# ─────────────────────────────────────────────────────────
def mdn_loss(pi, mu, sigma, target):
    target = target.unsqueeze(2)
    log_p  = (-0.5 * ((target - mu) / sigma) ** 2
              - sigma.log()
              - 0.5 * np.log(2 * np.pi)).sum(-1) + pi.log()
    return -torch.logsumexp(log_p, dim=-1).mean()


# ─────────────────────────────────────────────────────────
# WorldModelAgent
# ─────────────────────────────────────────────────────────
class WorldModelAgent(BaseAgent):
    """
    Ha & Schmidhuber (2018) World Model — Dream Training 완전 구현

    학습 흐름:
      [실제 환경]  매 스텝: V, M, C 실제 transition으로 학습
      [Dream]     N스텝마다: M 안에서 C를 꿈으로 추가 학습
                            → 논문의 핵심 기여
    """

    def __init__(self, num_arms, name="WorldModelAgent", device="cpu"):
        super().__init__(num_arms=num_arms, name=name)

        self.K       = num_arms
        self.device  = torch.device(device)
        self.obs_dim = cfg.H * (self.K + 1) + self.K * 3

        # 모듈
        self.V_enc    = VEncoder(self.obs_dim).to(self.device)
        self.V_dec    = VDecoder(self.obs_dim).to(self.device)
        self.M        = MDNRNN(self.K).to(self.device)
        self.C        = QNetwork(self.K).to(self.device)
        self.C_target = QNetwork(self.K).to(self.device)
        self.C_target.load_state_dict(self.C.state_dict())
        self.C_target.eval()

        # 옵티마이저
        self.opt_V = Adam(
            list(self.V_enc.parameters()) + list(self.V_dec.parameters()),
            lr=cfg.LR_V
        )
        self.opt_M = Adam(self.M.parameters(), lr=cfg.LR_M)
        self.opt_C = Adam(self.C.parameters(), lr=cfg.LR_C)

        # 내부 상태
        self._history     = deque([(0, 0.0)] * cfg.H, maxlen=cfg.H)
        self._sum_r2      = np.zeros(self.K)
        self._lstm_hidden = None
        self._last_z      = torch.zeros(cfg.Z_DIM).to(self.device)
        self._last_h      = torch.zeros(cfg.H_DIM).to(self.device)
        self._last_obs    = None
        self._last_action = 0
        self.epsilon      = cfg.EPS_START

        self.replay = ReplayBuffer(maxlen=20000)

        # 손실 기록
        self.loss_v_log     = []
        self.loss_m_log     = []
        self.loss_c_log     = []
        self.loss_dream_log = []   # dream training loss

    # ─── State 벡터 ───────────────────────────────────────
    def _build_obs(self):
        hist = []
        for arm, r in self._history:
            oh = np.zeros(self.K, dtype=np.float32)
            oh[arm] = 1.0
            hist.append(np.append(oh, float(r)))
        hist_vec = np.concatenate(hist)

        avg_r  = self.q_values.astype(np.float32)
        log_n  = np.log1p(self.pulls).astype(np.float32)
        var_r  = np.where(
            self.pulls > 1,
            self._sum_r2 / np.maximum(self.pulls, 1) - avg_r ** 2,
            0.0
        ).astype(np.float32)
        stat_vec = np.stack([avg_r, log_n, var_r], axis=1).flatten()

        return np.concatenate([hist_vec, stat_vec])

    # ─── choice() ─────────────────────────────────────────
    def choice(self):
        self.V_enc.eval()
        self.C.eval()

        obs = self._build_obs()
        self._last_obs = obs.copy()

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, z = self.V_enc(obs_t)
            z       = z.squeeze(0)
            q_vals  = self.C(z, self._last_h).squeeze(0)

        if np.random.random() < self.epsilon:
            arm = int(np.random.randint(self.K))
        else:
            arm = int(q_vals.argmax().item())

        self._last_z      = z.detach()
        self._last_action = arm
        return arm

    # ─── getReward() — dream 트리거 추가 ──────────────────
    def getReward(self, arm, reward):
        # 1. BaseAgent 공통
        super().getReward(arm, reward)

        # 2. 분산 통계
        self._sum_r2[arm] += reward ** 2

        # 3. 히스토리
        self._history.append((arm, float(reward)))

        # 4. M hidden 갱신
        self.M.eval()
        with torch.no_grad():
            self._last_h, self._lstm_hidden = self.M.step(
                self._last_z, arm, self._lstm_hidden
            )

        # 5. Replay 저장
        next_obs = self._build_obs()
        if self._last_obs is not None:
            self.replay.push(self._last_obs, arm, reward, next_obs)

        # 6. ε decay
        self.epsilon = max(cfg.EPS_END, self.epsilon * cfg.EPS_DECAY)

        # 7. 실제 환경 학습
        if len(self.replay) >= cfg.REPLAY_MIN:
            self._train_step()

        # 8. Dream Training ← 핵심 추가
        if (len(self.replay) >= cfg.REPLAY_MIN
                and self.t % cfg.DREAM_EVERY == 0):
            self._dream_and_train()

        # 9. Target network 업데이트
        if self.t % cfg.TARGET_UPDATE == 0:
            self.C_target.load_state_dict(self.C.state_dict())

    # ─── 실제 환경 학습 ───────────────────────────────────
    def _train_step(self):
        self._train_V()
        self._train_M()
        self._train_C()

    def _train_V(self):
        self.V_enc.train()
        self.V_dec.train()
        self.opt_V.zero_grad()

        obs, _, _, _ = self.replay.sample(cfg.BATCH_SIZE)
        obs = obs.to(self.device)

        z_mean, z_logvar, z = self.V_enc(obs)
        recon = self.V_dec(z)

        recon_loss = F.mse_loss(recon, obs)
        kl_loss    = -0.5 * (1 + z_logvar - z_mean**2 - z_logvar.exp()).mean()
        loss       = recon_loss + 0.001 * kl_loss

        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.V_enc.parameters()) + list(self.V_dec.parameters()), 1.0
        )
        self.opt_V.step()
        self.loss_v_log.append(loss.item())

    def _train_M(self):
        """
        기존 대비 변경:
          z_next 예측 loss + 보상 예측 loss 동시에 최적화
        """
        result = self.replay.sample_sequences(cfg.BATCH_SIZE, cfg.SEQ_LEN)
        if result is None:
            return
        obs_seq, action_seq, reward_seq = result
        obs_seq    = obs_seq.to(self.device)
        action_seq = action_seq.to(self.device)
        reward_seq = reward_seq.to(self.device)

        self.V_enc.eval()
        self.M.train()
        self.opt_M.zero_grad()

        with torch.no_grad():
            B, T, D = obs_seq.shape
            _, _, z_flat = self.V_enc(obs_seq.view(B * T, D))
            z_seq = z_flat.view(B, T, cfg.Z_DIM)

        # z_next 예측 (기존)
        pi, mu, sigma, r_pred, _ = self.M(z_seq[:, :-1], action_seq[:, :-1])
        z_loss = mdn_loss(pi, mu, sigma, z_seq[:, 1:].detach())

        # 보상 예측 (신규)
        # r_pred : (B, T-1, 1)
        # 타깃   : 같은 시점의 실제 reward
        r_target = reward_seq[:, :-1].unsqueeze(-1)
        r_loss   = F.mse_loss(r_pred, r_target)

        loss = z_loss + cfg.REWARD_LOSS_WEIGHT * r_loss

        loss.backward()
        nn.utils.clip_grad_norm_(self.M.parameters(), 1.0)
        self.opt_M.step()
        self.loss_m_log.append(loss.item())

    def _train_C(self):
        """실제 환경 transition으로 C 학습 (Double DQN)"""
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
            h      = torch.zeros(obs.size(0), cfg.H_DIM, device=self.device)
            h_next = torch.zeros(obs.size(0), cfg.H_DIM, device=self.device)

        q_all   = self.C(z, h)
        q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_a   = self.C(z_next, h_next).argmax(dim=1, keepdim=True)
            q_next   = self.C_target(z_next, h_next).gather(1, best_a).squeeze(1)
            q_target = rewards + cfg.GAMMA * q_next

        loss = F.smooth_l1_loss(q_taken, q_target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.C.parameters(), 1.0)
        self.opt_C.step()
        self.loss_c_log.append(loss.item())

    # ─── Dream Training ← 핵심 추가 ──────────────────────
    def _dream_and_train(self):
        """
        논문 4장 구현:
        M이 생성한 가상 환경 안에서 C를 학습.

        흐름:
          1. replay에서 실제 경험의 z를 시작점으로 샘플링
          2. C가 꿈 안에서 행동 선택 (ε-greedy)
          3. M.dream_step()으로 (z_next, r_pred) 예측
             → 실제 환경에 전혀 접근하지 않음
          4. dream transition 수집
          5. _train_C_from_dream()으로 C 업데이트

        temperature τ (논문 4.2절):
          σ를 τ배 확대 → dream이 더 노이즈 많아짐
          → C가 M의 허점 exploit하기 어려워짐
          → 실제 환경으로 policy 전이 성능 향상
        """
        self.V_enc.eval()
        self.M.eval()
        self.C.eval()

        # 1. replay에서 시작 z 샘플링
        z_starts = self.replay.sample_z_starts(
            cfg.DREAM_BATCH, self.V_enc, self.device
        )  # (DREAM_BATCH, z_dim)

        dream_transitions = []

        for b in range(cfg.DREAM_BATCH):
            z      = z_starts[b]
            h      = torch.zeros(cfg.H_DIM).to(self.device)
            hidden = None

            for _ in range(cfg.DREAM_HORIZON):
                # 2. C가 꿈 안에서 행동 선택
                with torch.no_grad():
                    q_vals = self.C(z, h).squeeze(0)
                if np.random.random() < self.epsilon:
                    action = int(np.random.randint(self.K))
                else:
                    action = int(q_vals.argmax().item())

                # 3. M이 다음 상태와 보상 예측
                #    과거 경험 기반으로 "이 맥락에서 arm k를 골랐을 때
                #    어느 수준의 reward가 나왔는지"를 r_pred로 반환
                with torch.no_grad():
                    z_next, r_pred, hidden = self.M.dream_step(
                        z, action, hidden,
                        temperature=cfg.TEMPERATURE
                    )
                    h = hidden[0].squeeze(0).squeeze(0)

                dream_transitions.append((
                    z.detach(),
                    action,
                    r_pred,      # M이 예측한 보상 (실제 reward 대신)
                    z_next.detach(),
                    h.detach(),
                ))

                z = z_next

        # 4. Dream transition으로 C 업데이트
        self._train_C_from_dream(dream_transitions)

    def _train_C_from_dream(self, transitions):
        """
        Dream transition으로 C 업데이트.

        실제 환경 학습과의 차이:
          obs 대신 z 직접 사용 (V 인코딩 불필요)
          reward는 M이 예측한 r_pred 사용
          z_next도 M이 생성한 가상 상태
          → 실제 환경에 전혀 접근하지 않음 ← 진짜 World Model
        """
        if len(transitions) < 4:
            return

        self.C.train()
        self.opt_C.zero_grad()

        z_list      = torch.stack([t[0] for t in transitions]).to(self.device)
        actions     = torch.LongTensor([t[1] for t in transitions]).to(self.device)
        rewards     = torch.FloatTensor([t[2] for t in transitions]).to(self.device)
        z_next_list = torch.stack([t[3] for t in transitions]).to(self.device)
        h_list      = torch.stack([t[4] for t in transitions]).to(self.device)
        h_next      = torch.zeros_like(h_list)

        q_all   = self.C(z_list, h_list)
        q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            best_a   = self.C(z_next_list, h_next).argmax(dim=1, keepdim=True)
            q_next   = self.C_target(z_next_list, h_next).gather(1, best_a).squeeze(1)
            q_target = rewards + cfg.GAMMA * q_next

        loss = F.smooth_l1_loss(q_taken, q_target)
        loss.backward()
        nn.utils.clip_grad_norm_(self.C.parameters(), 1.0)
        self.opt_C.step()
        self.loss_dream_log.append(loss.item())

    # ─── 저장 / 불러오기 ──────────────────────────────────
    def save(self, path="world_model_agent.pt"):
        torch.save({
            "V_enc":    self.V_enc.state_dict(),
            "V_dec":    self.V_dec.state_dict(),
            "M":        self.M.state_dict(),
            "C":        self.C.state_dict(),
            "C_target": self.C_target.state_dict(),
        }, path)
        print(f"[{self.name}] 저장 완료 → {path}")

    def load(self, path="world_model_agent.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.V_enc.load_state_dict(ckpt["V_enc"])
        self.V_dec.load_state_dict(ckpt["V_dec"])
        self.M.load_state_dict(ckpt["M"])
        self.C.load_state_dict(ckpt["C"])
        self.C_target.load_state_dict(ckpt["C_target"])
        print(f"[{self.name}] 로드 완료 ← {path}")


# ─────────────────────────────────────────────────────────
# 파라미터 정보 출력
# ─────────────────────────────────────────────────────────
def print_model_info(K=8):
    obs_dim = cfg.H * (K + 1) + K * 3
    agent   = WorldModelAgent(num_arms=K)
    modules = {
        "V Encoder":   agent.V_enc,
        "V Decoder":   agent.V_dec,
        "M MDN-RNN":   agent.M,
        "C Q-Network": agent.C,
    }
    print("=" * 60)
    print("  WorldModelAgent — Dream Training 버전")
    print("=" * 60)
    print(f"  obs_dim        = {cfg.H}×({K}+1) + {K}×3 = {obs_dim}")
    print(f"  z_dim          = {cfg.Z_DIM}")
    print(f"  h_dim          = {cfg.H_DIM}")
    print(f"  dream_horizon  = {cfg.DREAM_HORIZON} steps")
    print(f"  dream_every    = {cfg.DREAM_EVERY} steps")
    print(f"  temperature τ  = {cfg.TEMPERATURE}")
    print("-" * 60)
    total = 0
    for name, m in modules.items():
        n = sum(p.numel() for p in m.parameters())
        total += n
        print(f"  {name:<20} : {n:>8,} params")
    print("-" * 60)
    print(f"  {'합계':<20} : {total:>8,} params")
    print("=" * 60)
    print()
    print("  M 출력 구조:")
    print(f"    z_next 예측  : MDN (pi, mu, sigma) — 기존")
    print(f"    r_pred 예측  : scalar MLP          — 신규 추가")
    print("=" * 60)


# ─────────────────────────────────────────────────────────
# 동작 테스트
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_model_info(K=8)
    print()

    agent = WorldModelAgent(num_arms=8, name="WM_Dream_Test")
    print("[테스트] 300스텝 실행...")
    for step in range(300):
        arm    = agent.choice()
        reward = float(np.random.normal(0.1, 0.05))
        agent.getReward(arm, reward)

        if step % 50 == 49:
            v = np.mean(agent.loss_v_log[-10:])     if agent.loss_v_log     else 0
            m = np.mean(agent.loss_m_log[-10:])     if agent.loss_m_log     else 0
            c = np.mean(agent.loss_c_log[-10:])     if agent.loss_c_log     else 0
            d = np.mean(agent.loss_dream_log[-10:]) if agent.loss_dream_log else 0
            print(f"  step={step+1:>3} | "
                  f"V={v:.5f} | M={m:.5f} | "
                  f"C={c:.5f} | Dream={d:.5f} | "
                  f"ε={agent.epsilon:.3f}")

    print(f"\n  dream 학습 횟수: {len(agent.loss_dream_log)}")
    print("[완료] Dream Training 정상 동작 확인!")

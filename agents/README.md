# 🤖 Bandit Agents Collection

A collection of exploration-exploitation algorithms for multi-armed bandit and reinforcement learning environments. Implementations range from classic baselines to neural-enhanced adaptive agents.

---

## 📁 File Overview

### Classic Baselines

| File | Description |
|------|-------------|
| `ucb.py` | Standard UCB1 (Upper Confidence Bound). Selects actions by balancing empirical mean reward and an exploration bonus $\sqrt{\frac{2 \ln t}{n_i}}$. Theoretical regret bound: $O(\sqrt{KT \ln T})$. |
| `epsilon_greedy.py` | ε-Greedy policy. Exploits the best known action with probability $1-\varepsilon$, explores randomly otherwise. Simple but effective baseline. |
| `softmax.py` | Boltzmann (Softmax) exploration. Samples actions proportionally to $e^{Q_i / \tau}$ where $\tau$ is a temperature parameter controlling exploration intensity. |
| `thompson_sampling.py` | Thompson Sampling via Beta distribution posterior. Samples $\theta_i \sim \text{Beta}(\alpha_i, \beta_i)$ and acts greedily on the sample — Bayesian-optimal in many regimes. |
| `wsls.py` | Win-Stay Lose-Shift. Repeats the last action if it yielded a reward; switches otherwise. Memoryless and computationally lightweight. |

---

### Adaptive / Non-Stationary Variants

| File | Description |
|------|-------------|
| `decaying_epsilon.py` | ε-Greedy with schedule decay $\varepsilon_t = \varepsilon_0 / t^\alpha$. Gradually shifts from exploration to exploitation as data accumulates. |
| `sliding_window_ucb.py` | SW-UCB. Maintains a sliding window of the most recent $\tau$ observations per arm, discarding stale data. Designed for abruptly changing reward distributions. |
| `sw_decay_epsilon.py` | Sliding window + decaying epsilon hybrid. Combines recency-weighting with scheduled exploration decay for slowly drifting environments. |
| `periodic_ucb.py` | UCB with periodic forced exploration. Resets confidence bounds at fixed intervals to re-probe arms — useful when reward distributions change cyclically. |
| `thompson_weekly.py` | Thompson Sampling with periodic-reset posteriors. Resets Beta parameters every $N$ steps to adapt to periodic non-stationarity. |
| `thompson_collision_aware.py` | Collision-aware Thompson Sampling for multi-agent settings. Adjusts posterior updates when arm collisions are detected (multiple agents selecting the same arm simultaneously). |

---

### Feature-Aware / Contextual Agents

| File | Description |
|------|-------------|
| `as_ucb.py` | Attention-Score UCB. Computes a learned attention score over context features and folds it into the UCB exploration term, weighting arms by contextual relevance. |
| `fft_ucb.py` | FFT-UCB. Applies Fast Fourier Transform on reward history per arm to extract frequency-domain features (trend, seasonality), using spectral energy as a prior for the UCB bonus. |
| `sw_as_ucb.py` | Sliding Window + Attention Score UCB. Combines recency-limited observations with attention-weighted UCB bonuses for non-stationary contextual environments. |

---

### Neural / World Model Agents

| File | Description |
|------|-------------|
| `lstm_ucb.py` | LSTM-UCB. An LSTM encodes the sequence of (action, reward) history into a hidden state. UCB exploration is applied on top of the predicted Q-values. |
| `lstm_attention_ucb.py` | LSTM + Self-Attention + Cross-Attention UCB. Extends `lstm_ucb` with a self-attention mechanism over the LSTM output sequence and cross-attention between arm embeddings and the encoded context. Strongest model in the collection. |
| `world_model.py` | Latent World Model. Learns a compact latent representation of environment dynamics. Enables planning via imagined rollouts before committing to an action. |
)



---

## 📊 Recommended Usage by Environment Type

| Environment Characteristics | Recommended Agent |
|-----------------------------|-------------------|
| Stationary, no context | `ucb.py`, `thompson_sampling.py` |
| Abrupt distribution shifts | `sliding_window_ucb.py` |
| Gradual drift | `sw_decay_epsilon.py`, `decaying_epsilon.py` |
| Periodic / seasonal changes | `periodic_ucb.py`, `thompson_weekly.py` |
| Rich context features | `as_ucb.py`, `lstm_ucb.py` |
| Long-horizon temporal dependencies | `lstm_attention_ucb.py` |
| Multi-agent / collision setting | `thompson_collision_aware.py` |
| Model-based planning needed | `world_model.py` |

---


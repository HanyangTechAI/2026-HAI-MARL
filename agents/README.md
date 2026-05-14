# Bandit Agents 소개

다양한 탐색-활용(Exploration-Exploitation) 알고리즘 구현체 모음입니다. 고전적인 기본 모델부터 신경망 기반 적응형 에이전트까지 폭넓게 포함되어 있습니다.

---

## 📁 파일별 설명

### 고전 기본 모델 (Classic Baselines)

| 파일 | 설명 |
|------|------|
| `ucb.py` | 표준 UCB1 알고리즘. 경험적 평균 보상과 탐색 보너스 $\sqrt{\frac{2 \ln t}{n_i}}$ 를 합산해 행동을 선택합니다. 이론적 후회 한계: $O(\sqrt{KT \ln T})$. |
| `epsilon_greedy.py` | ε-탐욕 정책. 확률 $1-\varepsilon$ 로 현재 최선의 행동을 선택하고, 확률 $\varepsilon$ 로 무작위 탐색합니다. 단순하지만 강력한 기준 모델. |
| `softmax.py` | 볼츠만(Softmax) 탐색. 온도 파라미터 $\tau$ 를 이용해 $e^{Q_i / \tau}$ 에 비례하는 확률로 행동을 샘플링합니다. |
| `thompson_sampling.py` | 톰슨 샘플링. 각 팔마다 $\theta_i \sim \text{Beta}(\alpha_i, \beta_i)$ 에서 샘플링 후 탐욕적으로 행동 선택. 많은 환경에서 베이즈 최적에 수렴합니다. |
| `wsls.py` | Win-Stay Lose-Shift. 직전 행동이 보상을 받으면 유지, 그렇지 않으면 전환합니다. 메모리가 없고 연산이 가볍습니다. |

---

### 비정상 환경 적응 모델 (Adaptive / Non-Stationary)

| 파일 | 설명 |
|------|------|
| `decaying_epsilon.py` | 감쇠 ε-탐욕. $\varepsilon_t = \varepsilon_0 / t^\alpha$ 스케줄에 따라 탐색률이 점진적으로 감소합니다. |
| `sliding_window_ucb.py` | 슬라이딩 윈도우 UCB. 각 팔마다 최근 $\tau$ 스텝 관측값만 유지해 오래된 데이터를 버립니다. 보상 분포가 급변하는 환경에 적합합니다. |
| `sw_decay_epsilon.py` | 슬라이딩 윈도우 + 감쇠 ε 하이브리드. 최근성 가중치와 탐색 감쇠를 결합해 서서히 변화하는 환경에 대응합니다. |
| `periodic_ucb.py` | 주기적 강제 탐색 UCB. 고정 주기마다 신뢰 구간을 리셋해 팔을 재탐색합니다. 보상 분포가 주기적으로 변화하는 환경에 유용합니다. |
| `thompson_weekly.py` | 주간 리셋 톰슨 샘플링. 매 $N$ 스텝마다 Beta 사전 분포를 리셋해 주기적 비정상성에 적응합니다. |
| `thompson_collision_aware.py` | 충돌 인식 톰슨 샘플링. 다중 에이전트 환경에서 여러 에이전트가 동일한 팔을 선택할 때(충돌)를 감지하고 사후 분포 업데이트를 보정합니다. |

---

### 문맥 인식 모델 (Contextual / Feature-Aware)

| 파일 | 설명 |
|------|------|
| `as_ucb.py` | 어텐션 스코어 UCB. 문맥 특성에 대한 학습된 어텐션 점수를 UCB 탐색 항에 통합해 문맥적으로 관련 있는 팔에 가중치를 부여합니다. |
| `fft_ucb.py` | FFT-UCB. 각 팔의 보상 이력에 고속 푸리에 변환(FFT)을 적용해 주파수 도메인 특성(추세, 계절성)을 추출하고, 스펙트럼 에너지를 UCB 보너스의 사전 정보로 활용합니다. |
| `sw_as_ucb.py` | 슬라이딩 윈도우 + 어텐션 스코어 UCB. 최근 관측값 제한과 어텐션 가중치 UCB 보너스를 결합해 비정상 문맥 환경에 대응합니다. |

---

### 신경망 / 세계 모델 에이전트 (Neural / World Model)

| 파일 | 설명 |
|------|------|
| `lstm_ucb.py` | LSTM-UCB. LSTM이 (행동, 보상) 이력 시퀀스를 은닉 상태로 인코딩합니다. 예측된 Q값 위에 UCB 탐색을 적용합니다. |
| `lstm_attention_ucb.py` | LSTM + Self-Attention + Cross-Attention UCB. LSTM 출력 시퀀스에 셀프 어텐션을, 팔 임베딩과 인코딩된 문맥 사이에 크로스 어텐션을 추가한 최상위 모델입니다. |
| `world_model.py` | 잠재 세계 모델. 환경 동역학의 압축된 잠재 표현을 학습합니다. 행동을 결정하기 전 상상된 롤아웃을 통한 플래닝이 가능합니다. |

---

## 📊 환경 특성별 우세 에이전트

| 환경 특성 | 추천 에이전트 |
|-----------|--------------|
| 정상(Stationary), 문맥 없음 | `ucb.py`, `thompson_sampling.py` |
| 보상 분포 급변 | `sliding_window_ucb.py` |
| 보상 분포 완만한 변화 | `sw_decay_epsilon.py`, `decaying_epsilon.py` |
| 주기적 / 계절적 변화 | `periodic_ucb.py`, `thompson_weekly.py` |
| 풍부한 문맥 특성 | `as_ucb.py`, `lstm_ucb.py` |
| 장기 시간적 의존성 | `lstm_attention_ucb.py` |
| 다중 에이전트 / 충돌 환경 | `thompson_collision_aware.py` |
| 모델 기반 플래닝 필요 | `world_model.py` |

---

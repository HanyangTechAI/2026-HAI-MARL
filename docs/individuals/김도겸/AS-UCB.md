#### 1. AS-UCB (Adaptive Seasonal UCB)

**배경 및 목적 (Motivation)**
기존의 UCB 알고리즘은 월마트 데이터처럼 '주말 효과(Weekly Seasonality)'가 뚜렷한 환경에서 주말의 매출 급등을 단순한 '분산(Noise)'으로 취급하는 치명적인 한계가 있었다. 이를 해결하기 위해, 매장의 **'기본 체력(Global Mean)'**과 특정 요일의 **'주기적 오차(Seasonal Offset)'**를 분리하여 학습하는 AS-UCB를 고안하였다. 초기 데이터 부족으로 인한 과대적합(Early Overfitting)을 방지하기 위해 베이지안 스무딩(Bayesian Smoothing) 기법을 도입하였다.

**수학적 공식 (Mathematical Formulation)**
특정 매장 $a$에 대한 스텝 $t$에서의 가치 함수 $Q_t(a)$는 다음과 같이 정의된다.

$$Q_t(a) = \hat{\mu}_a + \tilde{\Delta}_a(p_t) + c \sqrt{\frac{\ln t}{N_a}}$$

* $\hat{\mu}_a$: 전체 기간의 글로벌 평균 (베이스라인)
* $p_t$: 현재 스텝의 주기 위상 (예: $t \pmod 7$)
* $\tilde{\Delta}_a(p_t)$: 베이지안 스무딩이 적용된 요일별 오차 가감치
  $$\tilde{\Delta}_a(p_t) = \frac{\sum_{i=1}^{N_{a, p_t}} (R_{a, i} - \hat{\mu}_{a, i})}{N_{a, p_t} + \lambda}$$
  (단, $\lambda$는 초기 노이즈를 억제하는 스무딩 파라미터)

**한계점 (Limitation)**
순수한 파동형(Shock) 환경에서는 우수하나, 장기적으로 우상향/우하향하는 **트렌드(Trend) 환경에서 알고리즘이 붕괴**하는 현상이 발견되었다. 글로벌 평균이 과거에 머물러 있기 때문에, 트렌드에 의한 상승분을 '주기적 효과'로 심각하게 착각(Misinterpretation)하는 약점이 노출되었다.
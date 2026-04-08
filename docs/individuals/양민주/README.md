# 양민주 - Thompson Sampling 실험

## 파일 설명

### 알고리즘 구현 파일

1. **thompson_sampling.py** - 기본 Thompson Sampling
   - Beta 분포를 사용한 베이지안 접근
   - reward_scale 파라미터로 보상 스케일 조정
   - 모든 시점을 하나의 Beta 분포로 학습

2. **thompson_weekly.py** - 주말/평일 분리 Thompson
   - 주말(토,일)과 평일(월~금)을 별도 Beta 분포로 학습
   - 월마트 데이터의 주기적 패턴 활용
   - 타이밍 차별화로 충돌 회피

3. **thompson_collision_aware.py** - 충돌 인식 Thompson
   - 실제 보상이 기대보다 낮으면 충돌로 판단
   - 충돌 arm에 패널티 부여하여 회피
   - 과밀 환경(8~10명)에서 안정적 성능

### 실험 파일

- **experiment_marl.py** - MARL 환경 실험 코드
  - 4개 실험 (6명, 7명, 8명, 10명 경쟁)
  - 다양한 Thompson 변형 비교
  - Epsilon-Greedy, UCB와 성능 비교

### 실험 보고서

- **20260408.md** - Thompson Sampling 실험 결과
  - 주말/평일 분리, 충돌 인식 전략 실험
  - 7명 경쟁에서 1위 달성
  - 10명 경쟁에서 2위, 3위 달성

## 주요 성과

- Thompson_x7: 7명 경쟁에서 **1위** (249.37)
- Thompson_Weekend: 7명 경쟁에서 **3위** (236.96)
- Thompson_CollisionAware: 10명 경쟁에서 **2위** (201.16)

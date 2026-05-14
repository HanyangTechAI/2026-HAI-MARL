# 🛒 Walmart M5 Data Processing for MARL Environment

## 📚: Description
Real-world financial and retail markets aren't just random dice rolls. Sales spike on weekends, and demand explodes exponentially during holidays like Black Friday.

It contains the data processing and feature extraction pipeline designed for Multi-Agent Reinforcement Learning simulator. To ensure our Ai agents train in a highly realistic "Non-stationary Environment", this pipeline extracts the market's core features from 5 years(1,941 days) of Walmart sales records.

---

## 📂 File Structure& Core Guide

### 1. 📊 Processed Environmental Data
* `sales_train_evaluation.csv`
  * 5년(1,941일) 간의 캘리포니아(CA), 텍사스(TX), 위스콘신(WI) 지역 일일 판매량(Unit Sales) 원본 데이터.
* `calendar.csv`
  * 각 스텝(d_1 ~ d_1941)의 실제 날짜 정보, 요일, 그리고 주요 이벤트(SuperBowl, Thanksgiving 등) 정보가 담긴 메타 데이터.

### 2. ⚙️ 데이터 전처리 및 특징 추출 스크립트 (Scripts)
* `data_extractor.py`
  * 원본 데이터에서 각 주(State) 및 중분류(Dept)별 '평균(Mean)', '분산(Variance)', '우상향 기울기(Trend)' 등 정적(Stationary) 기초 파라미터를 추출하는 스크립트.
* `feature_extractor.py` **[핵심 파이프라인]**
  * 시계열 분해(STL)와 IQR 기법을 사용하여, 데이터 속 숨겨진 '요일별 주기성(Seasonality)' 파동과 '돌발 충격(Shock)' 배수를 정밀하게 추출해 내는 자동화 스크립트. 

### 3. 📁 생성된 파라미터 보관소 (Output)
* `extracted_data/` 
  * `feature_extractor.py`를 실행하면 생성되는 최종 파라미터 보관 폴더입니다. 메인 시뮬레이터(`main_time.py`)가 이곳의 데이터를 읽어 환경을 조립합니다.
  * `seasonality_registry.csv`: 각 물류 라인(Arm)의 요일별(0~6) 매출 파동 계수(Multiplier).
  * `shocks_registry.csv`: 통계적으로 유의미한 폭증/폭락이 발생한 특정 스텝과 그 충격 배수(Multiplier).

### 4. 📈 시각화 자료 (Visualizations)
* `CA_21_dept_trends.png` / `TX_21_dept_trends.png` / `WI_21_dept_trends.png`
  * 각 주(State)별 7개 중분류(Dept)의 5년 치 매출 흐름을 30일 이동평균(30-Day Moving Average)으로 시각화한 그래프. 시뮬레이터에 투입할 15개 자산(Arm) 후보군의 개성을 파악할 때 참고합니다.

---

## 🚀 환경 데이터 업데이트 (실행 방법)

만약 충격(Shock) 감지 기준(Threshold)을 변경하거나 새로운 매대의 데이터를 뽑아내고 싶다면, 이 폴더 위치에서 다음 스크립트를 순서대로 실행하세요.

```bash
# 1. 21개 중분류의 시각화 및 정적 데이터(평균/기울기) 확인
python data_extractor.py

# 2. 다이내믹 환경(파동+충격) 추출 및 CSV 갱신 (시뮬레이터 반영)
python feature_extractor.py

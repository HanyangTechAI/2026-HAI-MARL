import pandas as pd
import numpy as np
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import linregress

def extract_all_features():
    # ==========================================
    # 📍 동적 경로 설정 (스크립트 위치 기준)
    # ==========================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sales_path = os.path.join(script_dir, "sales_train_evaluation.csv")
    output_dir = os.path.join(script_dir, "extracted_data")
    
    if not os.path.exists(sales_path):
        print(f"🚨 에러: '{sales_path}' 파일을 찾을 수 없습니다!")
        return

    print("⏳ [특징 추출] 21개 전체 매대의 파라미터, 충격(Shock), 주기(Seasonality) 분석 중...")
    os.makedirs(output_dir, exist_ok=True)
        
    sales = pd.read_csv(sales_path)
    
    base_params_list = []
    shocks_list = []
    seasonality_list = []

    # 3개 State와 7개 Dept 자동 인식
    states = ['CA', 'TX', 'WI']
    depts = sales['dept_id'].unique()
    x_axis = np.arange(1941)

    # ==========================================
    # 🎯 21개 전체 매대 순회 (Cross Loop)
    # ==========================================
    for state in states:
        for dept in depts:
            arm_name = f"{state}_{dept}"
            
            # 해당 state와 dept의 데이터만 필터링하여 일일 판매량 합산
            target_data = sales[(sales['state_id'] == state) & (sales['dept_id'] == dept)]
            series = target_data.loc[:, 'd_1':'d_1941'].sum()
            
            # ------------------------------------------------
            # 📊 1. 기초 파라미터 & 트렌드(기울기) 추출
            # ------------------------------------------------
            y_values = series.values
            slope, intercept, _, _, _ = linregress(x_axis, y_values)
            predicted_y = intercept + slope * x_axis
            variance = np.var(y_values - predicted_y) # 회귀선 대비 잔차 분산
            
            # 나중에 어떤 Arm 클래스로 조립하든 쓸 수 있게 모든 값을 저장
            base_params_list.append({
                'arm_name': arm_name,
                'base_mean': round(series.mean(), 2), # 일반 평균 (Stationary, Shock 용)
                'start_mean': round(intercept, 2),    # 트렌드 시작점 (Trend 용)
                'slope': round(slope, 4),             # 트렌드 기울기
                'base_variance': round(variance, 2)
            })
            
            # ------------------------------------------------
            # 📅 2. 주기성(Seasonality) 분석
            # ------------------------------------------------
            series_df = pd.DataFrame({'sales': series.values})
            series_df['day_of_week'] = series_df.index % 7 
            baseline_mean = series.mean()
            
            # 판매량이 0인 유령 매대가 아니면 계산
            if baseline_mean > 0:
                weekly_pattern = series_df.groupby('day_of_week')['sales'].mean() / baseline_mean
                for dow, mult in weekly_pattern.items():
                    seasonality_list.append({'arm_name': arm_name, 'day_of_week': dow, 'multiplier': round(mult, 3)})

            # ------------------------------------------------
            # ⚡ 3. 이벤트 충격(Shock) 분석
            # ------------------------------------------------
            # 평균 판매량이 10개 미만인 매대는 노이즈가 너무 심해 STL 분해가 무의미하므로 패스
            if baseline_mean > 10:
                result = seasonal_decompose(series, model='additive', period=7)
                baseline_series = result.trend + result.seasonal
                residuals = result.resid.dropna()
                
                Q1, Q3 = residuals.quantile(0.25), residuals.quantile(0.75)
                IQR = Q3 - Q1
                STRICTNESS = 1.5 
                upper, lower = Q3 + (STRICTNESS * IQR), Q1 - (STRICTNESS * IQR)
                anomalies = residuals[(residuals > upper) | (residuals < lower)]
                
                for step_idx, _ in anomalies.items():
                    step_num = int(step_idx.replace('d_', ''))
                    expected = baseline_series[step_idx]
                    actual = series[step_idx]
                    
                    if expected > 0:
                        mult = round(actual / expected, 2)
                        # 예상치 대비 15% 이상 변동이 있는 것만 충격으로 기록
                        if mult >= 1.15 or mult <= 0.85:
                            shock_type = "MEGA" if (mult >= 1.5 or mult <= 0.5) else "MICRO"
                            shocks_list.append({
                                'arm_name': arm_name, 
                                'step': step_num, 
                                'multiplier': mult,
                                'type': shock_type
                            })

    # ==========================================
    # 💾 CSV로 완벽하게 3장 저장!
    # ==========================================
    pd.DataFrame(base_params_list).to_csv(os.path.join(output_dir, "base_params_registry.csv"), index=False)
    pd.DataFrame(shocks_list).to_csv(os.path.join(output_dir, "shocks_registry.csv"), index=False)
    pd.DataFrame(seasonality_list).to_csv(os.path.join(output_dir, "seasonality_registry.csv"), index=False)
    
    print(f"✅ 추출 완료! 21개 모든 매대의 정보가 '{output_dir}' 폴더에 CSV 파일 3개로 완벽 저장되었습니다.")

if __name__ == "__main__":
    extract_all_features()
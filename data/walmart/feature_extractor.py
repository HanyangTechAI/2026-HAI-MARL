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

    print("⏳ [특징 추출] 기초 파라미터, 충격(Shock), 주기(Seasonality) 분석 중...")
    os.makedirs(output_dir, exist_ok=True)
        
    sales = pd.read_csv(sales_path)
    
    base_params_list = []
    shocks_list = []
    seasonality_list = []

    # ==========================================
    # 🎯 1. 다이내믹 주식 (FOODS_3) 추출 파트
    # ==========================================
    foods_3 = sales[sales['dept_id'] == 'FOODS_3']
    daily_sales_foods = foods_3.groupby('state_id').sum().loc[:, 'd_1':'d_1941']

    for state in ['CA', 'TX', 'WI']:
        series = daily_sales_foods.loc[state]
        arm_name = f"{state}_FOODS_3"
        
        # 📊 [추가됨] Base Mean & Variance 추출
        base_params_list.append({
            'arm_name': arm_name,
            'arm_type': 'EventShock',
            'base_mean': round(series.mean(), 2),
            'base_variance': round(series.var(), 2),
            'slope': 0.0 # FOODS는 트렌드보단 주기가 강하므로 기울기 0 처리
        })
        
        # --- 주기성(Seasonality) 분석 ---
        series_df = pd.DataFrame({'sales': series.values})
        series_df['day_of_week'] = series_df.index % 7 
        baseline_mean = series.mean()
        weekly_pattern = series_df.groupby('day_of_week')['sales'].mean() / baseline_mean
        
        for dow, mult in weekly_pattern.items():
            seasonality_list.append({'arm_name': arm_name, 'day_of_week': dow, 'multiplier': round(mult, 3)})

        # --- 충격(Shock) 분석 ---
        result = seasonal_decompose(series, model='additive', period=7)
        baseline = result.trend + result.seasonal
        residuals = result.resid.dropna()
        
        Q1, Q3 = residuals.quantile(0.25), residuals.quantile(0.75)
        IQR = Q3 - Q1
        STRICTNESS = 1.5 
        upper, lower = Q3 + (STRICTNESS * IQR), Q1 - (STRICTNESS * IQR)
        anomalies = residuals[(residuals > upper) | (residuals < lower)]
        
        for step_idx, _ in anomalies.items():
            step_num = int(step_idx.replace('d_', ''))
            expected = baseline[step_idx]
            actual = series[step_idx]
            
            if expected > 0:
                mult = round(actual / expected, 2)
                if mult >= 1.15 or mult <= 0.85:
                    shock_type = "MEGA" if (mult >= 1.5 or mult <= 0.5) else "MICRO"
                    shocks_list.append({
                        'arm_name': arm_name, 
                        'step': step_num, 
                        'multiplier': mult,
                        'type': shock_type
                    })

    # ==========================================
    # 🎯 2. 장기 트렌드 성장주 (HOUSEHOLD_1) 추출 파트
    # ==========================================
    household_1 = sales[sales['dept_id'] == 'HOUSEHOLD_1']
    daily_sales_hh = household_1.groupby('state_id').sum().loc[:, 'd_1':'d_1941']
    x_axis = np.arange(1941)

    for state in ['CA', 'TX', 'WI']:
        y_values = daily_sales_hh.loc[state].values
        arm_name = f"{state}_HOUSEHOLD_1"
        
        # 📈 선형 회귀로 시작점(intercept)과 기울기(slope) 추출
        slope, intercept, _, _, _ = linregress(x_axis, y_values)
        predicted_y = intercept + slope * x_axis
        variance = np.var(y_values - predicted_y) # 회귀선 대비 잔차 분산
        
        base_params_list.append({
            'arm_name': arm_name,
            'arm_type': 'Trend',
            'base_mean': round(intercept, 2), # TrendArm의 start_mean 역할
            'base_variance': round(variance, 2),
            'slope': round(slope, 4)
        })

    # ==========================================
    # 💾 CSV로 완벽하게 3장 저장!
    # ==========================================
    pd.DataFrame(base_params_list).to_csv(os.path.join(output_dir, "base_params_registry.csv"), index=False)
    pd.DataFrame(shocks_list).to_csv(os.path.join(output_dir, "shocks_registry.csv"), index=False)
    pd.DataFrame(seasonality_list).to_csv(os.path.join(output_dir, "seasonality_registry.csv"), index=False)
    
    print(f"✅ 추출 완료! '{output_dir}' 폴더에 CSV 파일 3개가 안전하게 저장되었습니다.")

if __name__ == "__main__":
    extract_all_features()
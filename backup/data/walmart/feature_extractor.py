import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import os

def extract_all_features(sales_path, output_dir="extracted_data"):
    print("⏳ [특징 추출] 충격(Shock)과 주기(Seasonality) 분석 중...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    sales = pd.read_csv(sales_path)
    foods_3 = sales[sales['dept_id'] == 'FOODS_3']
    daily_sales = foods_3.groupby('state_id').sum().loc[:, 'd_1':'d_1941']
    
    shocks_list = []
    seasonality_list = []

    for state in ['CA', 'TX', 'WI']:
        series = daily_sales.loc[state]
        
        # --- 1. 주기성(Seasonality) 분석 ---
        # 요일별(0~6) 평균 Multiplier를 계산합니다. (예: 주말은 1.3배, 월요일은 0.8배)
        # d_1이 토요일(Saturday)부터 시작한다는 가정하에 t % 7 로 요일 매핑
        series_df = pd.DataFrame({'sales': series.values})
        series_df['day_of_week'] = series_df.index % 7 
        
        baseline_mean = series.mean()
        weekly_pattern = series_df.groupby('day_of_week')['sales'].mean() / baseline_mean
        
        for dow, mult in weekly_pattern.items():
            seasonality_list.append({'arm_name': f"{state}_FOODS_3", 'day_of_week': dow, 'multiplier': round(mult, 3)})

        # --- 2. 충격(Shock) 분석 (기준 완화 및 등급화) ---
        result = seasonal_decompose(series, model='additive', period=7)
        baseline = result.trend + result.seasonal
        residuals = result.resid.dropna()
        
        Q1, Q3 = residuals.quantile(0.25), residuals.quantile(0.75)
        IQR = Q3 - Q1
        
        # 🎛️ 밸브 1: IQR 배수를 2.5에서 1.5로 낮춰서 더 많은 이상치를 포착!
        STRICTNESS = 1.5 
        upper, lower = Q3 + (STRICTNESS * IQR), Q1 - (STRICTNESS * IQR)
        
        anomalies = residuals[(residuals > upper) | (residuals < lower)]
        
        for step_idx, _ in anomalies.items():
            step_num = int(step_idx.replace('d_', ''))
            expected = baseline[step_idx]
            actual = series[step_idx]
            
            if expected > 0:
                mult = round(actual / expected, 2)
                
                # 🎛️ 밸브 2: 15% 이상 변동하면 모두 충격으로 인정!
                if mult >= 1.15 or mult <= 0.85:
                    
                    # 🎛️ 밸브 3: 충격의 크기에 따라 '등급(Type)' 부여
                    if mult >= 1.5 or mult <= 0.5:
                        shock_type = "MEGA"   # 50% 이상 폭발/폭락 (슈퍼볼 급)
                    else:
                        shock_type = "MICRO"  # 15~49% 변동 (소규모 행사 급)
                        
                    shocks_list.append({
                        'arm_name': f"{state}_FOODS_3", 
                        'step': step_num, 
                        'multiplier': mult,
                        'type': shock_type # 등급 정보 추가
                    })

    # CSV로 완벽하게 저장!
    pd.DataFrame(shocks_list).to_csv(f"{output_dir}/shocks_registry.csv", index=False)
    pd.DataFrame(seasonality_list).to_csv(f"{output_dir}/seasonality_registry.csv", index=False)
    print(f"✅ 추출 완료! '{output_dir}' 폴더에 CSV 파일 2개가 저장되었습니다.")

if __name__ == "__main__":
    extract_all_features("sales_train_evaluation.csv")
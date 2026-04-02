# arms/arm_registry.py

"""
월마트 실제 데이터 기반 파라미터 보관소 (16-Arm Master Roster)
- Monte Carlo 시뮬레이션 시 각 카테고리에서 2개씩 랜덤 차출하여 8개의 시장을 구성합니다.
"""

STATIONARY_REGISTRY = {
    "CA_HOBBIES_1": {"mean": 1482.4, "variance": 86278.62},
    "WI_HOBBIES_1": {"mean": 715.02, "variance": 30718.47},
    "TX_FOODS_1":   {"mean": 675.19, "variance": 28419.22},
    "WI_HOUSEHOLD_2": {"mean": 334.65, "variance": 5960.63}
}

EVENT_SHOCK_REGISTRY = {
    "WI_FOODS_3": {"base_mean": 4725.38, "base_variance": 1261774.75}, # 🍯 독이 든 성배 (함정)
    "CA_FOODS_2": {"base_mean": 1555.18, "base_variance": 139561.97},
    "CA_FOODS_1": {"base_mean": 1242.03, "base_variance": 75377.29},
    "TX_FOODS_2": {"base_mean": 1134.98, "base_variance": 65754.61}
}

TREND_REGISTRY = {
    "CA_HOUSEHOLD_1": {"start_mean": 1568.21, "slope": 1.0285, "variance": 375637.94},
    "TX_HOUSEHOLD_1": {"start_mean": 1211.36, "slope": 0.6768, "variance": 184853.98},
    "WI_HOUSEHOLD_1": {"start_mean": 989.69,  "slope": 0.6353, "variance": 176595.24},
    "WI_FOODS_2":     {"start_mean": 594.47,  "slope": 0.7540, "variance": 182229.05}
}

SWITCH_REGISTRY = {
    "CA_HOUSEHOLD_2": {"base_mean": 816.61, "base_variance": 30144.75},
    "WI_FOODS_1":     {"base_mean": 756.87, "base_variance": 33666.01},
    "TX_HOBBIES_1":   {"base_mean": 738.70, "base_variance": 24388.71},
    "TX_HOUSEHOLD_2": {"base_mean": 415.59, "base_variance": 7434.18}
}
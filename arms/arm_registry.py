# arms/arm_registry.py

"""
월마트 실제 데이터 기반 파라미터 보관소 (Golden Balance Edition)
- 모든 자산이 1100 ~ 1800 사이의 경쟁력 있는 평균을 가집니다.
- 초반, 중반, 후반의 승자가 모두 다른 다이내믹한 밸런스입니다.
"""

STATIONARY_REGISTRY = {
    "CA_HOBBIES_1": {
        "description": "캘리포니아 취미용품1 (탄탄한 중위권 피난처)",
        "mean": 1482.4,        
        "variance": 86278.62    
    },
    "CA_FOODS_1": {
        "description": "캘리포니아 식품1 (무난하고 안정적임)",
        "mean": 1242.03,
        "variance": 75377.29    
    },
}

EVENT_SHOCK_REGISTRY = {
    "CA_FOODS_2": {
        "description": "캘리포니아 식품2 (적절한 수익과 잦은 주말 파동)",
        "base_mean": 1555.18,
        "base_variance": 139561.97  
    },
    "TX_FOODS_2": {
        "description": "텍사스 식품2 (다이내믹 2등주)",
        "base_mean": 1134.98,
        "base_variance": 65754.61
    }
}

TREND_REGISTRY = {
    "CA_HOUSEHOLD_1": {
        "description": "캘리포니아 생활용품1 (후반 지배형 성장주)",
        "start_mean": 1568.21,
        "slope": 1.0285,       # 시뮬레이션 종료 시 약 3500 도달
        "variance": 375637.94
    },
    "WI_FOODS_2": {
        "description": "위스콘신 식품2 (꼴찌에서 시작해 1등을 노리는 대기만성주)",
        "start_mean": 594.47,
        "slope": 0.7540,       # 시뮬레이션 종료 시 약 2000 도달
        "variance": 182229.05
    }
}

SWITCH_REGISTRY = {
    "TX_HOUSEHOLD_1": {
        "description": "텍사스 생활용품1 (초반 1등주 -> 특정 기점 대폭락)",
        "base_mean": 1867.86,
        "base_variance": 184853.98
    },
    "WI_HOUSEHOLD_1": {
        "description": "위스콘신 생활용품1 (초반 대장주 -> 특정 기점 국면 전환)",
        "base_mean": 1605.93,
        "base_variance": 176595.24
    }
}
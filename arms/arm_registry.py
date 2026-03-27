# arms/arm_registry.py

"""
월마트 데이터 파라미터 보관소.
"""
STATIONARY_REGISTRY = {
    "CA_FOODS_1": {
        "description": "캘리포니아 신선식품 (가장 수요가 많고 안정적)",
        "mean": 3500.0,
        "variance": 500.0
    },
    "TX_FOODS_1": {
        "description": "텍사스 신선식품 (수요 높음, 변동성 약간 있음)",
        "mean": 2800.0,
        "variance": 450.0
    },
    "WI_FOODS_1": {
        "description": "위스콘신 신선식품 (수요 중간, 매우 안정적)",
        "mean": 2100.0,
        "variance": 300.0
    },
    "CA_HOUSEHOLD_1": {
        "description": "캘리포니아 생활용품 (수요 적음, 꾸준함)",
        "mean": 1200.0,
        "variance": 150.0
    }
}

EVENT_SHOCK_REGISTRY = {
    "CA_FOODS_3": {
        "description": "캘리포니아 스낵/파티류 (주말 파동 + 슈퍼볼 등 충격 극심)",
        "base_mean": 7267.6,
        "base_variance": 1200.0
    },
    "TX_FOODS_3": {
        "description": "텍사스 스낵/파티류 (주말 파동 + 명절 충격)",
        "base_mean": 4976.1,
        "base_variance": 800.0
    },
    "WI_FOODS_3": {
        "description": "위스콘신 스낵/파티류 (가장 변동성 큼)",
        "base_mean": 4725.4,
        "base_variance": 800.0
    }
}
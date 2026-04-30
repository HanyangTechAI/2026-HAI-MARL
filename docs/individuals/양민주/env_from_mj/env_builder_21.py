# docs/individuals/양민주/env_builder_21.py
"""
21개 Arm 실험용 환경 빌더
"""
import os
import random
from envs.SMPyBandits.Environment.MAB import MAB
from arms.stationary_arm import StationaryArm
from arms.event_shock_arm import EventShockArm
from arms.trend_arm import TrendArm
from arms.switch_arm import SwitchArm
from docs.individuals.양민주.arm_registry_21 import (
    STATIONARY_REGISTRY, EVENT_SHOCK_REGISTRY, TREND_REGISTRY, SWITCH_REGISTRY
)

# 상수 경로
SHOCKS_FILE = os.path.join("data", "walmart", "extracted_data", "shocks_registry.csv")
SEASON_FILE = os.path.join("data", "walmart", "extracted_data", "seasonality_registry.csv")
SWITCH_FILE = os.path.join("data", "walmart", "switched_data", "regime_switches.csv")

def build_dynamic_market_21(strategy="balanced", num_arms=21):
    """21개 arm 실험용 환경 빌더
    
    Args:
        strategy: 차출 전략 ("balanced", "random", etc.)
        num_arms: 차출할 arm의 총 개수 (기본값: 21, 최대 21)
    """
    sampled_arms = {"stationary": [], "shocks": [], "trends": [], "switches": []}
    
    if strategy == "balanced":
        # 각 카테고리에서 균등하게 차출
        per_category = num_arms // 4
        remainder = num_arms % 4
        
        counts = [per_category] * 4
        for i in range(remainder):
            counts[i] += 1
        
        sampled_arms["stationary"] = random.sample(
            list(STATIONARY_REGISTRY.keys()), 
            min(counts[0], len(STATIONARY_REGISTRY))
        )
        sampled_arms["shocks"] = random.sample(
            list(EVENT_SHOCK_REGISTRY.keys()), 
            min(counts[1], len(EVENT_SHOCK_REGISTRY))
        )
        sampled_arms["trends"] = random.sample(
            list(TREND_REGISTRY.keys()), 
            min(counts[2], len(TREND_REGISTRY))
        )
        sampled_arms["switches"] = random.sample(
            list(SWITCH_REGISTRY.keys()), 
            min(counts[3], len(SWITCH_REGISTRY))
        )
    else:
        all_keys = (
            [("stationary", k) for k in STATIONARY_REGISTRY.keys()] +
            [("shocks", k) for k in EVENT_SHOCK_REGISTRY.keys()] +
            [("trends", k) for k in TREND_REGISTRY.keys()] +
            [("switches", k) for k in SWITCH_REGISTRY.keys()]
        )
        for cat, name in random.sample(all_keys, min(num_arms, len(all_keys))):
            sampled_arms[cat].append(name)
            
    arm_configuration = []
    arm_names = []
    
    for name in sampled_arms["stationary"]:
        p = STATIONARY_REGISTRY[name]
        arm_configuration.append(StationaryArm(name, p["mean"], p["variance"]))
        arm_names.append(name)
        
    for name in sampled_arms["shocks"]:
        p = EVENT_SHOCK_REGISTRY[name]
        arm_configuration.append(EventShockArm(name, p["base_mean"], p["base_variance"], SHOCKS_FILE, SEASON_FILE))
        arm_names.append(name)
        
    for name in sampled_arms["trends"]:
        p = TREND_REGISTRY[name]
        arm_configuration.append(TrendArm(name, p["start_mean"], p["slope"], p["variance"]))
        arm_names.append(name)
        
    for name in sampled_arms["switches"]:
        p = SWITCH_REGISTRY[name]
        arm_configuration.append(SwitchArm(name, p["base_mean"], p["base_variance"], SWITCH_FILE))
        arm_names.append(name)
        
    return MAB(arm_configuration), arm_names

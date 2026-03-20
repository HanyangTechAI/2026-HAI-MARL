# SMPyBandits Core
이 폴더의 모듈들은 오픈소스 다중 에이전트 MAB 라이브러리인 [SMPyBandits](https://github.com/SMPyBandits/SMPyBandits)에서 우리 프로젝트(2026-HAI-MARL-Bandits)에 필요한 핵심 환경(Environment) 코드만 발췌하여 수정한 것입니다.

무거운 통신 로직을 제거하고, 해당 프로젝트에 필요한 모듈들만 사용할 예정입니다.

## Source & License
- **Original Repository:** [https://github.com/SMPyBandits/SMPyBandits](https://github.com/SMPyBandits/SMPyBandits)
- **License:** MIT License (Copyright (c) 2016-2018 Lilian Besson et al.)

## 포함된 핵심 모듈 설명
- `Arm.py`, `Gaussian.py`, `RestlessArm.py`: 자산 수익률 확률 분포를 모사하는 팔(Arm) 객체
- `MAB.py`: 여러 팔을 묶어 관리하는 카지노(시장) 환경 객체
- `CollisionModels.py`: 다중 에이전트 충돌 시 보상 분배 모델 참고용
- `EvaluatorMultiPlayers.py` 등: 평가기 로직 참고용 (직접 호출하지 않으며 핵심 로직만 `custom_evaluator.py`로 이식)
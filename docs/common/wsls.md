# 🧠 WSLS (Win-Stay, Lose-Shift) 에이전트 명세서

## 1. 개요 (Overview)
WSLS 에이전트는 전통적인 강화학습의 '수학적 최적화'를 모사하는 대신, 행동경제학과 진화생물학에서 증명된 **인간의 인지 편향과 감정적 휴리스틱(Heuristic)**을 모사하는 알고리즘입니다. 

자신의 '기대 수익(Aspiration Level)'을 기준으로 삼아, 어제 번 돈이 기대 이상이면 만족하여 그 자리에 머물고(Win-Stay), 기대 이하면 분노하여 다른 무작위 자산으로 갈아타는(Lose-Shift) 극단적이고 근시안적인 행동 패턴을 보입니다.

## 2. 핵심 메커니즘: 동적 기대 수익 (Dynamic Aspiration Level)
인간은 고정된 목표를 가지지 않고 환경에 적응합니다. 어제까지 1,000달러를 기대하던 사람도 하락장을 겪으면 "500달러만 벌어도 다행이다"라며 눈높이를 낮춥니다. 이를 수식으로 구현한 것이 **동적 열망 수준 업데이트**입니다.

* **업데이트 수식:**
  `A(t+1) = (1 - lambda) * A(t) + lambda * R(t)`
  * `A(t)`: 현재 스텝의 기대 수익 (Aspiration Level)
  * `R(t)`: 실제 획득한 보상 (Reward)
  * `lambda`: 기대치 적응률 (Aspiration Learning Rate)

## 3. 하이퍼파라미터 (Hyperparameters)
WSLS 클래스를 인스턴스화할 때 다음 두 가지 파라미터로 에이전트의 '성격'을 조절할 수 있습니다.

| 파라미터 | 타입 | 기본값 | 설명 | 성격 모사 |
| :--- | :--- | :--- | :--- | :--- |
| `initial_aspiration` | float | 0.5 | 시뮬레이션 시작 시 에이전트가 마음속에 품고 있는 최초의 목표 수익입니다. | 높을수록 초반에 '대장주병'에 걸려 방황함 |
| `aspiration_lr` | float | 0.05 | 수식의 `lambda` 값. 현실의 보상 결과에 자신의 눈높이를 얼마나 빨리 맞출 것인지 결정합니다. | `0.01`(똥고집, 타협 안 함) ~ `0.2`(팔랑귀, 금방 순응함) |

## 4. 월마트 다이내믹 환경에서의 관전 포인트
* **슬리피지(Slippage) 지옥에서의 생존력:** 남들이 몰려 수익이 깎이는 순간(Lose) 즉시 다른 곳으로 튀기(Shift) 때문에, 멍청해 보이지만 의외로 슬리피지를 잘 회피하는 생존력을 보일 수 있습니다.
* **충격(Shock)에 대한 취약성:** 대장주에서 단 하루 이벤트로 파동이 쳐서 수익이 일시적으로 깎이면, 펀더멘탈을 보지 않고 바로 손절(Shift)을 치기 때문에 장기적인 거래 기회비용을 엄청나게 지불하게 됩니다.

## 5. 참고 문헌 (References)
* Nowak, M., & Sigmund, K. (1993). A strategy of win-stay, lose-shift that outperforms tit-for-tat in the Prisoner's Dilemma game. *Nature*.
* Bendor, J., Diermeier, D., & Ting, M. (2001). A Behavioral Model of Turnout. *American Political Science Review*.
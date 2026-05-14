# Comparative Analysis of Adaptive MARL Strategies in Non-Stationary Financial Markets

## 💻 Description
This project is a comprehensive Multi-Agent Reinforcement Learning (MARL) testbed designed to analyze the "Density Paradox" - the phenomenon where multiple AI agents compete for limited resources in a highly volatile, non-stationary financial/retail market. It aims to identify the most robust survival and profit-maximizing strategies in overcrowded environments.

### 🔑 Key Features
- Dynamic Market Modeling: Simulates 4 distinct market regimes - Stationary, Shock, Trend and Regime Switch - using real-world Walmart retail data (M5 Dataset).
- Slippage Penalty: Implements a non-linear penalty for overcrowded assets to replicate real-world "Market Impact" and penalize irrational "Herd Behavior".
- Algorithm Battle Royale: A massive competitive ecosystem featuring over 57 different RL algorithms, ranging from human-like heuristics and classical statistics to complex RL algorithms.
- TrueSkill Evaluation: Introduces an objective, relative rating system based on TrueSkill to evaluate the true strategic robustness of agents amidst severe market noise.

## Conclusion

### Summary of Experimental Results
- **Derivation of the Best Algorithm**: Experimental results showed that the exponential decay-based **Decaying Epsilon-Greedy** algorithm achieved the best performance, recording the highest TrueSkill Score (≈89.7) and a top 30% survival rate (≈79.6%).
- **Demonstration of Strategic Robustness**: This suggests that in environments with extreme uncertainty, bold initial exploration and an adaptive exploration-exploitation transition mechanism are more effective in securing strategic robustness than complex pattern learning or fixed heuristics.

### Project Significance
- **Strategy Verification in a Complex Non-stationary Market**: This study established a complex non-stationary market environment based on real retail data, where regime switches and slippage penalties coexist, and empirically verified the adaptability and survivability of multi-agent reinforcement learning strategies.
- **Introduction of a Relative Evaluation Framework**: To overcome the limitations of absolute profit metrics, this study introduced a relative evaluation framework based on TrueSkill to compare the practical superiority among agents amidst market noise.

### Future Research Plans
- **Integration with Advanced Learning Models**: It is expected that by enhancing the capability to respond to non-stationarity through future integration with deep learning and meta-learning, more advanced multi-agent decision-making outcomes can be generated.

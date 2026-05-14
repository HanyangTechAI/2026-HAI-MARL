# Comparative Analysis of Adaptive MARL Strategies in Non-Stationary Financial Markets

## 💻 Description
This project is a comprehensive Multi-Agent Reinforcement Learning (MARL) testbed designed to analyze the "Density Paradox" - the phenomenon where multiple AI agents compete for limited resources in a highly volatile, non-stationary financial/retail market. It aims to identify the most robust survival and profit-maximizing strategies in overcrowded environments.

### 🔑 Key Features
- Dynamic Market Modeling: Simulates 4 distinct market regimes - Stationary, Shock, Trend and Regime Switch - using real-world Walmart retail data (M5 Dataset).
- Slippage Penalty: Implements a non-linear penalty for overcrowded assets to replicate real-world "Market Impact" and penalize irrational "Herd Behavior".
- Algorithm Battle Royale: A massive competitive ecosystem featuring over 57 different RL algorithms, ranging from human-like heuristics and classical statistics to complex RL algorithms.
- TrueSkill Evaluation: Introduces an objective, relative rating system based on TrueSkill to evaluate the true strategic robustness of agents amidst severe market noise.

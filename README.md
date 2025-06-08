# Regime-Sensitive Risk Modeling with HMMs and EWMA

## Project Overview

The aim of this project is to build a **regime-sensitive risk monitoring system** that dynamically adapts to changing market conditions using **Hidden Markov Models (HMMs)** and **Exponentially Weighted Moving Averages (EWMA)**. It targets improved risk estimation and portfolio allocation by capturing shifts between hidden market regimes like bull, bear, or high-volatility states.

---

## Objectives

- To detect latent market regimes from historical asset returns using HMM.
- Compute regime-specific risk metrics including Value-at-Risk (VaR).
- Apply EWMA to calculate regime-conditioned covariance matrices.
- Demonstrate superior risk control over static models via regime-aware VaR tracking.

---

## Literature Basis

- **Ang & Bekaert (2003)** – Showed regime-switching models improve international asset allocation by adjusting to changing volatilities and correlations.
- **Giudici & Abu Hashish (2020)** – Applied HMMs to detect regime changes in cryptoasset markets, confirming their practical use in high-volatility domains.
- **Sharma & Krishna (2023)** – Proposed an aggregate Markov-based reliability model, validating the use of Markov chains in dynamic environments with multiple operational states.

---

## Methodology

1. **Data Acquisition**
   - Gathering daily prices across multiple assets through sources like Yahoo Finance, Quandl, or Bloomberg.
   - Then compute log returns and construct a time-indexed matrix.

2. **Regime Detection**
   - Train a Gaussian HMM (e.g., with `hmmlearn`) on the return series.
   - Assign each time point a hidden market state (regime) using the Viterbi path.

3. **Covariance Estimation**
   - Compute EWMA-based correlation matrices for each regime.
   - Convert these to covariance matrices using asset volatilities.

4. **VaR Calculation**
   - For a given portfolio and confidence level, compute 1-day VaR per regime using:  
     ```
     VaR_α^(k) = z_α * sqrt(wᵀ * Σ^(k) * w)
     ```

5. **Visualization & Analysis**
   - We then plot regime-labeled time series, rolling VaR curves, and regime-specific correlation heatmaps.

---

## Tools & Libraries

- `Python`, `pandas`, `numpy`, `hmmlearn`, `matplotlib`, `seaborn`
- Optional: `plotly`, `streamlit`, `scikit-learn` for interactive visualization

---

## Results Preview

- The model detects significant regime shifts (e.g., pre/post-crash).
- Adjusts risk metrics dynamically to reflect volatility.
- Improves VaR coverage by adapting to changing market states.

---

## Potential Research Extensions

- Add macroeconomic variables (e.g., inflation, VIX) as features in the HMM.
- Build hierarchical HMMs for cross-asset regime modeling.
- Combine with reinforcement learning to rebalance portfolios across detected regimes.

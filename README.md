# Shuttlecock Hitting Event Detection & Rally Analysis in Badminton (CV + ML)

This repository contains the implementation for a research project on **shuttlecock hitting event detection** and **rally analysis** in badminton using **computer vision** and **machine learning**.  
The pipeline extracts hit events from match videos, constructs **hit-windows** to handle timing uncertainty, engineers **baseline (F0)** and **enhanced (F1)** rally features, and evaluates **winner prediction** models using **leak-safe GroupKFold** splitting by video.

---

## Project Overview

**Goal:** Automatically analyze badminton match videos by:
1. Detecting **shuttlecock hit events**
2. Segmenting rallies and extracting rally indicators
3. Engineering match-level feature sets (**F0** and **F1**)
4. Training/evaluating ML models to predict the **match winner**

---

## Key Contributions

- **Hit event → Hit-window sampling**: Constructs temporal windows around predicted hit frames (offset `k ∈ {0, 3, 5}`) to increase robustness under small timing shifts.
- **Rally analytics pipeline**: Converts hit events into rally-level indicators (tempo, rally length, inferred zones/landing proxies, etc.).
- **Feature engineering**:
  - **F0 (Baseline)**: fundamental hit/rally statistics
  - **F1 (Enhanced)**: adds domain-informed spatial/tempo/distribution features
- **Winner prediction**: Compares models (e.g., Logistic Regression / Random Forest / SVM / XGBoost*) and feature sets (F0 vs F1 vs F0+F1).
- **Leak-safe evaluation**: Uses **GroupKFold by VideoName** to prevent train/test leakage across the same match video.

\* XGBoost is optional depending on environment.

---

## Results Highlights (Update with your final numbers)

- **Hit-window shuttle visibility**:
  - `k=0`: 39.62%
  - `k=±3`: 64.90%
  - `k=±5`: 74.16%
- **Winner prediction**:
  - Best model: **[Fill in: e.g., Random Forest + F1]**
  - Accuracy / Macro-F1: **[Fill in your final mean scores]**
  - Tuning improved RF accuracy (example): **0.6816 → 0.6916** (update if needed)


---

## Repository Structure


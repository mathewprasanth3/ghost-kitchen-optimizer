# 🍽️ Ghost Kitchen Demand Optimizer
AI/ML system for demand prediction across location–cuisine combinations, 
combining clustering, XGBoost forecasting, and SHAP explainability.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://ghostkitchendashboard.streamlit.app)

---

## 📊 Results
| Metric | Score |
|--------|-------|
| Accuracy | 78.2% |
| Recall (High-Demand) | 77% |
| Class Imbalance Handled | 2.35:1 via scale_pos_weight |
| Location-Cuisine Combinations | 5,000+ |

---

## 🧠 Overview
Predicts demand across location–cuisine combinations to support 
data-driven decisions for cloud kitchen expansion. Uses KMeans to 
segment markets and XGBoost to classify high vs low demand locations, 
with SHAP for business-interpretable explanations.

---

## ⚙️ Features
- **Market Segmentation (KMeans)** — clusters locations by demand patterns
- **Demand Prediction (XGBoost)** — 78.2% accuracy with class balancing
- **Explainability (SHAP)** — feature importance for business decisions
- **Interactive Dashboard (Streamlit)** — real-time predictions

---

## 🏗️ Tech Stack
Python, XGBoost, Scikit-learn, SHAP, Pandas, NumPy, Streamlit

---

## 👤 Author
Mathew Prasanth, P.E. | AI/ML Engineer
[LinkedIn](https://www.linkedin.com/in/mathewprasanth/)

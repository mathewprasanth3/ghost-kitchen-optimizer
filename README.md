# 🍽️ Ghost Kitchen Demand Optimizer

AI/ML system for demand prediction across location–cuisine combinations, combining clustering, XGBoost forecasting, and SHAP explainability.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red)](https://ghostkitchendashboard.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-green)](https://xgboost.readthedocs.io)

---

## 📊 Results

| Metric | Value |
|---|---|
| Accuracy | 78.2% |
| Recall (High-Demand Markets) | 77% |
| Class Imbalance | 2.35:1 — handled via scale_pos_weight |
| Location-Cuisine Combinations | 5,000+ |

---

## 🧠 What It Does

Identifying high-demand ghost kitchen locations requires analysing hundreds of location-cuisine combinations simultaneously. This system automates that analysis — segmenting markets with KMeans, predicting demand with XGBoost, and explaining predictions with SHAP so operators can trust and act on the output.

The pipeline performs three tasks:
- Segments locations into market clusters based on demand patterns
- Classifies high vs low demand for each location-cuisine combination
- Explains which features drove each prediction via SHAP

This reflects real-world market intelligence workflows where interpretability is as important as accuracy — operators need to understand *why* a location is flagged before committing capital.

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Segmentation | KMeans clustering |
| Prediction | XGBoost classifier |
| Explainability | SHAP (feature importance + summary plots) |
| Dashboard | Streamlit |
| Data | Zomato restaurant dataset |

---

## 🏗️ Key Training Decisions

- `scale_pos_weight=2.35` — handles 2.35:1 class imbalance, boosting recall on minority high-demand class
- KMeans segmentation before classification — reduces noise by grouping similar markets first
- SHAP TreeExplainer — model-native explainability, no approximation overhead
- 80/20 stratified train/test split — preserves class ratio across sets

---

## 📁 Project Structure

```
ghost-kitchen-optimizer/
├── app/
│   └── ghost_kitchen_dashboard.py
├── data/
│   └── zomato.csv
├── models/
│   ├── demand_predictor.pkl
│   └── shap_explainer.pkl
├── src/
│   ├── clustering.py
│   ├── demand_predictor.py
│   └── price_recommender.py
├── notebooks/
│   └── 01_data_exploration.ipynb
└── main.py
```

---

## 🚀 Run Locally

```bash
git clone https://github.com/mathewprasanth/ghost-kitchen-optimizer.git
cd ghost-kitchen-optimizer
pip install -r requirements.txt
streamlit run app/ghost_kitchen_dashboard.py
```

---

## ⚠️ Limitations

- Dataset is Zomato India — predictions are India-specific
- Demand proxy derived from ratings and competition, not actual order volume
- Multi-label cuisine combinations not yet supported

---

## 👤 Author

**Mathew Prasanth, P.E.**
AI/ML Engineer | U.S. Licensed Professional Engineer
[LinkedIn](https://www.linkedin.com/in/mathewprasanth/) · [Live Demo](https://ghostkitchendashboard.streamlit.app)

*AWS Certified ML Specialty · AWS Cloud Practitioner*

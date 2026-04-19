import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os

# ------------------------------
# 1. Load Data
# ------------------------------
df = pd.read_csv("data/combo_output_with_clusters.csv")

# ------------------------------
# 2. Build Lookup Tables
# ------------------------------
# Used at prediction time to fill in features
# for any location + cuisine combination the user enters

location_avg_competition = df.groupby("location")["competition"].mean().to_dict()
cuisine_avg_rating       = df.groupby("cuisine")["rating"].mean().to_dict()
cuisine_avg_price        = df.groupby("cuisine")["price"].mean().to_dict()

# Global fallbacks for completely unseen values
global_avg_competition = df["competition"].mean()
global_avg_rating      = df["rating"].mean()
global_avg_price       = df["price"].mean()

print("📊 Lookup tables built:")
print(f"  Locations: {len(location_avg_competition)}")
print(f"  Cuisines:  {len(cuisine_avg_rating)}")

# ------------------------------
# 3. Build Training Features from Lookups
# ------------------------------
# Training uses the same lookup logic so the model learns
# from aggregated signals — not individual row values
df["rating_input"]      = df["cuisine"].map(cuisine_avg_rating)
df["competition_input"] = df["location"].map(location_avg_competition)
df["price_input"]       = df["cuisine"].map(cuisine_avg_price)

# ------------------------------
# 4. Create Target — High Demand
# ------------------------------
# Target is real customer behavior (votes/demand)
# NOT derived from the opportunity score formula
threshold = df["demand"].quantile(0.70)
df["high_demand"] = (df["demand"] >= threshold).astype(int)

print(f"\nDemand threshold (top 30%): {threshold:.2f}")
print(f"Class distribution:\n{df['high_demand'].value_counts()}")

# ------------------------------
# 5. Encode Cuisine
# ------------------------------
le_cuisine = LabelEncoder()
df["cuisine_encoded"] = le_cuisine.fit_transform(df["cuisine"])

# ------------------------------
# 6. Features & Target
# ------------------------------
# All features derivable from just location + cuisine input
# No demand used as input — genuine prediction
features = ["rating_input", "competition_input", "price_input", "cuisine_encoded"]
X = df[features]
y = df["high_demand"]

print(f"\nFeatures: {features}")
print(f"Total samples: {len(X)}")

# ------------------------------
# 7. Train / Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------
# 8. Train XGBoost
# ------------------------------
model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# ------------------------------
# 9. Cross-Validation
# ------------------------------
cv_scores = cross_val_score(model, X, y, cv=5, scoring="f1")
print(f"\n📊 5-Fold CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ------------------------------
# 10. Evaluate
# ------------------------------
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Low Demand", "High Demand"]))
print("🔢 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ------------------------------
# 11. Feature Importance
# ------------------------------
print("\n📌 Feature importances:")
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.3f}")

# ------------------------------
# 12. Save Everything
# ------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model,                    "models/demand_predictor.pkl")
joblib.dump(le_cuisine,               "models/label_encoder_cuisine.pkl")
joblib.dump(features,                 "models/demand_features.pkl")
joblib.dump(threshold,                "models/demand_threshold.pkl")
joblib.dump(location_avg_competition, "models/location_avg_competition.pkl")
joblib.dump(cuisine_avg_rating,       "models/cuisine_avg_rating.pkl")
joblib.dump(cuisine_avg_price,        "models/cuisine_avg_price.pkl")
joblib.dump({
    "competition": global_avg_competition,
    "rating":      global_avg_rating,
    "price":       global_avg_price
},                                    "models/global_fallbacks.pkl")
print("\n✅ All artifacts saved to models/")

# ------------------------------
# 13. SHAP
# ------------------------------
print("\n🔍 Running SHAP analysis...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

os.makedirs("models/plots", exist_ok=True)
plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=features, show=False)
plt.tight_layout()
plt.savefig("models/plots/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(shap_values, X_test, feature_names=features, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("models/plots/shap_importance.png", dpi=150, bbox_inches="tight")
plt.close()

joblib.dump(explainer, "models/shap_explainer.pkl")
print("✅ SHAP saved")

# ------------------------------
# 14. Sanity Check
# ------------------------------
print("\n🧪 Sanity check — Sushi in Indiranagar (may not exist in dataset):")
loc  = "Indiranagar"
cui  = "Sushi"
comp  = location_avg_competition.get(loc, global_avg_competition)
rat   = cuisine_avg_rating.get(cui, global_avg_rating)
price = cuisine_avg_price.get(cui, global_avg_price)
cenc  = le_cuisine.transform([cui])[0] if cui in le_cuisine.classes_ else -1
inp   = pd.DataFrame([[rat, comp, price, cenc]], columns=features)
pred  = model.predict(inp)[0]
proba = model.predict_proba(inp)[0][1]
print(f"  → {'High Demand ✅' if pred == 1 else 'Low Demand ⚠️'} ({proba*100:.1f}% confidence)")

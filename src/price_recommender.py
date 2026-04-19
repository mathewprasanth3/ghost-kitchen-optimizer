import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import os

# ------------------------------
# 1. Load Data
# ------------------------------
df = pd.read_csv("data/combo_output_with_clusters.csv")

# ------------------------------
# 2. Create Price Bucket Target
# ------------------------------
# Budget: bottom 33%, Mid: middle 33%, Premium: top 33%
df["price_bucket"] = pd.qcut(
    df["price"],
    q=3,
    labels=["Budget", "Mid", "Premium"]
)

print("💰 Price Bucket Distribution:")
print(df["price_bucket"].value_counts())
print(f"\nPrice ranges:")
print(df.groupby("price_bucket", observed=False)["price"].agg(["min", "max"]))

# ------------------------------
# 3. Build Lookup Tables (same approach as demand predictor)
# ------------------------------
# These allow inference for unseen location-cuisine combinations
location_avg_competition    = df.groupby("location")["competition"].mean().to_dict()
location_avg_demand         = df.groupby("location")["demand"].mean().to_dict()
location_avg_normalized     = df.groupby("location")["normalized_demand"].mean().to_dict()
cuisine_avg_rating          = df.groupby("cuisine")["rating"].mean().to_dict()

# Global fallbacks for completely unseen values
global_avg_competition  = df["competition"].mean()
global_avg_demand       = df["demand"].mean()
global_avg_normalized   = df["normalized_demand"].mean()
global_avg_rating       = df["rating"].mean()

print(f"\n📊 Lookup tables built:")
print(f"  Locations: {len(location_avg_competition)}")
print(f"  Cuisines:  {len(cuisine_avg_rating)}")

# ------------------------------
# 4. Build Training Features from Lookups
# ------------------------------
# Same approach as demand_predictor — train on aggregated signals
# so inference works for any location-cuisine combination
df["demand_input"]      = df["location"].map(location_avg_demand)
df["rating_input"]      = df["cuisine"].map(cuisine_avg_rating)
df["competition_input"] = df["location"].map(location_avg_competition)
df["norm_demand_input"] = df["location"].map(location_avg_normalized)

# ------------------------------
# 5. Encode Cuisine & Target
# ------------------------------
# Load the same cuisine encoder used by demand_predictor for consistency
le_cuisine = joblib.load("models/label_encoder_cuisine.pkl") \
    if os.path.exists("models/label_encoder_cuisine.pkl") \
    else LabelEncoder().fit(df["cuisine"])

df["cuisine_encoded"] = le_cuisine.transform(df["cuisine"])

le_price = LabelEncoder()
df["price_bucket_encoded"] = le_price.fit_transform(df["price_bucket"])

# ------------------------------
# 6. Features & Target
# ------------------------------
# All features derivable from just location + cuisine — same as demand predictor
features = ["demand_input", "rating_input", "competition_input", "norm_demand_input", "cuisine_encoded"]
X = df[features]
y = df["price_bucket_encoded"]

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
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="mlogloss",
    num_class=3
)
model.fit(X_train, y_train)

# ------------------------------
# 9. Evaluate
# ------------------------------
y_pred = model.predict(X_test)
print("\n📊 Price Recommender Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=le_price.classes_
))

# ------------------------------
# 10. Save Model & Artifacts
# ------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(model,                  "models/price_recommender.pkl")
joblib.dump(le_price,               "models/label_encoder_price.pkl")
joblib.dump(features,               "models/price_features.pkl")
joblib.dump(location_avg_competition, "models/price_location_avg_competition.pkl")
joblib.dump(location_avg_demand,    "models/price_location_avg_demand.pkl")
joblib.dump(location_avg_normalized,"models/price_location_avg_normalized.pkl")
joblib.dump(cuisine_avg_rating,     "models/price_cuisine_avg_rating.pkl")
joblib.dump({
    "competition":    global_avg_competition,
    "demand":         global_avg_demand,
    "normalized":     global_avg_normalized,
    "rating":         global_avg_rating,
},                                  "models/price_global_fallbacks.pkl")

# Save price bucket ranges for display in Streamlit
price_ranges = df.groupby("price_bucket", observed=False)["price"].agg(["min", "max"]).to_dict()
joblib.dump(price_ranges, "models/price_ranges.pkl")

print("\n✅ Price recommender saved to models/price_recommender.pkl")
print("✅ All lookup tables saved to models/")

# ------------------------------
# 11. Sanity Check — unseen combination
# ------------------------------
print("\n🧪 Sanity check — Korean BBQ in Banjara Hills (may not exist in dataset):")
loc = "Banjara Hills"
cui = "Korean"

comp    = location_avg_competition.get(loc, global_avg_competition)
demand  = location_avg_demand.get(loc, global_avg_demand)
norm    = location_avg_normalized.get(loc, global_avg_normalized)
rat     = cuisine_avg_rating.get(cui, global_avg_rating)
cenc    = le_cuisine.transform([cui])[0] if cui in le_cuisine.classes_ else -1

inp  = pd.DataFrame([[demand, rat, comp, norm, cenc]], columns=features)
pred = le_price.inverse_transform(model.predict(inp))[0]
print(f"  → Recommended price tier: {pred}")

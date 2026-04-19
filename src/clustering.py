import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------
# 1. Load Data
# -------------------------------
df = pd.read_csv("data/combo_output.csv")

# -------------------------------
# 2. Fix Invalid Ratings (IMPORTANT)
# -------------------------------
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

# Remove invalid ratings (0 or NaN)
df = df[df["rating"] > 0]

# OPTIONAL: Remove very low-quality/noisy data
df = df[df["rating"] >= 2.0]

# -------------------------------
# 3. Validate Required Columns
# -------------------------------
required_columns = ["normalized_demand", "competition", "rating"]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# -------------------------------
# 4. Handle Missing Values
# -------------------------------
df = df.dropna(subset=required_columns)

# -------------------------------
# 5. Feature Selection
# -------------------------------
features = ["normalized_demand", "competition", "rating"]
X = df[features]

# -------------------------------
# 6. Feature Scaling (CRITICAL)
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 7. Apply KMeans Clustering
# -------------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(X_scaled)

# -------------------------------
# 8. Analyze Clusters
# -------------------------------
print("\n📊 Cluster Summary:\n")
cluster_summary = df.groupby("cluster")[features].mean()
print(cluster_summary)

# -------------------------------
# 9. Assign Business Labels (FINAL)
# -------------------------------
cluster_names = {
    0: "Emerging Market 🌱",   # low competition, decent rating
    1: "Low Potential ❌",     # weak demand + poor rating
    2: "Saturated ⚔️"         # high demand + high competition
}

df["cluster_name"] = df["cluster"].map(cluster_names)

# -------------------------------
# 10. Save Output
# -------------------------------
output_path = "data/combo_output_with_clusters.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Clustering complete. Saved to: {output_path}")

# -------------------------------
# 11. Debug: Rating Distribution
# -------------------------------
print("\n📈 Rating Distribution Check:\n")
print(df["rating"].describe())
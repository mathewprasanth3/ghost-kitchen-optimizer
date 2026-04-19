import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import os

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Ghost Kitchen Optimizer", layout="wide")

col1, col2 = st.columns([3, 1])
with col1:
    st.title("🍽️ Ghost Kitchen Demand Optimizer")
    st.markdown("*Data-driven cuisine recommendations for ghost kitchen launches across India*")
with col2:
    st.markdown("###")  # Keeps your top alignment spacer
    st.markdown(
        "**Mathew Prasanth, PE**\n\n"
        "AI/ML Engineer"
    )
st.divider()

# ------------------------------
# Load Data & Models
# ------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(base_dir, "..", "data", "combo_output_with_clusters.csv"))

@st.cache_resource
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    m = os.path.join(base_dir, "..", "models")
    return {
        # Demand predictor
        "demand_model":     joblib.load(os.path.join(m, "demand_predictor.pkl")),
        "demand_features":  joblib.load(os.path.join(m, "demand_features.pkl")),
        "demand_loc_comp":  joblib.load(os.path.join(m, "location_avg_competition.pkl")),
        "demand_cui_rat":   joblib.load(os.path.join(m, "cuisine_avg_rating.pkl")),
        "demand_cui_price": joblib.load(os.path.join(m, "cuisine_avg_price.pkl")),
        "demand_fallbacks": joblib.load(os.path.join(m, "global_fallbacks.pkl")),

        # Price recommender
        "price_model":      joblib.load(os.path.join(m, "price_recommender.pkl")),
        "price_features":   joblib.load(os.path.join(m, "price_features.pkl")),
        "price_loc_comp":   joblib.load(os.path.join(m, "price_location_avg_competition.pkl")),
        "price_loc_dem":    joblib.load(os.path.join(m, "price_location_avg_demand.pkl")),
        "price_loc_norm":   joblib.load(os.path.join(m, "price_location_avg_normalized.pkl")),
        "price_cui_rat":    joblib.load(os.path.join(m, "price_cuisine_avg_rating.pkl")),
        "price_fallbacks":  joblib.load(os.path.join(m, "price_global_fallbacks.pkl")),
        "price_ranges":     joblib.load(os.path.join(m, "price_ranges.pkl")),

        # Shared
        "le_cuisine":       joblib.load(os.path.join(m, "label_encoder_cuisine.pkl")),
        "le_price":         joblib.load(os.path.join(m, "label_encoder_price.pkl")),
        "explainer":        joblib.load(os.path.join(m, "shap_explainer.pkl")),
    }

df  = load_data()
mdl = load_models()

CLUSTER_COLORS = {
    "Emerging Market 🌱": "#2ecc71",
    "Low Potential ❌":   "#e74c3c",
    "Saturated ⚔️":      "#e67e22"
}
CLUSTER_INFO = {
    "Emerging Market 🌱": "Low competition + decent ratings. Great entry opportunity.",
    "Low Potential ❌":   "Weak demand and poor ratings. Not recommended.",
    "Saturated ⚔️":      "High demand but heavy competition. Risky entry."
}

# ------------------------------
# Helper — encode cuisine
# ------------------------------
def encode_cuisine(cuisine):
    try:
        return mdl["le_cuisine"].transform([cuisine])[0]
    except ValueError:
        return -1

# ------------------------------
# Helper — demand prediction via lookup tables
# FIXED: confidence now reflects the predicted class, not always High Demand
# ------------------------------
def predict_demand(location, cuisine):
    cenc  = encode_cuisine(cuisine)
    comp  = mdl["demand_loc_comp"].get(location, mdl["demand_fallbacks"]["competition"])
    rat   = mdl["demand_cui_rat"].get(cuisine,   mdl["demand_fallbacks"]["rating"])
    price = mdl["demand_cui_price"].get(cuisine,  mdl["demand_fallbacks"]["price"])

    inp        = pd.DataFrame([[rat, comp, price, cenc]], columns=mdl["demand_features"])
    pred       = mdl["demand_model"].predict(inp)[0]
    proba      = mdl["demand_model"].predict_proba(inp)[0]
    proba_high = proba[1]
    proba_low  = proba[0]

    # Show confidence for whichever class was predicted
    conf = round((proba_high if pred == 1 else proba_low) * 100, 1)

    return pred, conf, inp

# ------------------------------
# Helper — price prediction via lookup tables
# ------------------------------
def predict_price(location, cuisine):
    cenc   = encode_cuisine(cuisine)
    comp   = mdl["price_loc_comp"].get(location, mdl["price_fallbacks"]["competition"])
    demand = mdl["price_loc_dem"].get(location,  mdl["price_fallbacks"]["demand"])
    norm   = mdl["price_loc_norm"].get(location, mdl["price_fallbacks"]["normalized"])
    rat    = mdl["price_cui_rat"].get(cuisine,   mdl["price_fallbacks"]["rating"])

    inp    = pd.DataFrame([[demand, rat, comp, norm, cenc]], columns=mdl["price_features"])
    bucket = mdl["le_price"].inverse_transform(mdl["price_model"].predict(inp))[0]
    return bucket

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("🔧 Filters")
locations = sorted(df["location"].unique())
selected_location = st.sidebar.selectbox("📍 Select Location", locations)
all_clusters = sorted(df["cluster_name"].unique())
selected_clusters = st.sidebar.multiselect("🏷️ Filter by Cluster", all_clusters, default=all_clusters)
top_n = st.sidebar.slider("🔢 Top N Cuisines", 1, 10, 5)

st.sidebar.divider()
st.sidebar.markdown("### 🗺️ Cluster Legend")
for label, desc in CLUSTER_INFO.items():
    color = CLUSTER_COLORS.get(label, "#888")
    st.sidebar.markdown(
        f"<span style='color:{color}; font-weight:bold'>{label}</span><br><small>{desc}</small>",
        unsafe_allow_html=True
    )

st.sidebar.divider()
st.sidebar.markdown("### 🤖 About the Models")
st.sidebar.markdown(
    "**Demand Predictor** — predicts high/low demand using average competition, "
    "rating, and price for the location-cuisine combination. Works for unseen combos.\n\n"
    "**Price Recommender** — recommends Budget/Mid/Premium using the same lookup approach. "
    "Both models use lookup table averages so any combination can be predicted."
)

# ------------------------------
# App Mode
# ------------------------------
mode = st.radio("Choose Mode", ["🔍 Explore a Location", "🎯 Predict Any Combination"], horizontal=True)
st.divider()

# ============================================================
# MODE 1 — Explore a Location
# ============================================================
if mode == "🔍 Explore a Location":

    st.subheader("🔍 Explore a Location")
    st.markdown("Select a location to see top cuisines ranked by opportunity score with ML demand forecasts.")

    filtered_df = df[
        (df["location"] == selected_location) &
        (df["cluster_name"].isin(selected_clusters))
    ].copy()

    filtered_df["custom_score"] = (
        0.5 * filtered_df["normalized_demand"] +
        0.3 * filtered_df["rating"] -
        0.2 * filtered_df["competition"]
    ).round(3)

    top_df = filtered_df.sort_values("custom_score", ascending=False).head(top_n).copy()

    if top_df.empty:
        st.warning("No results for selected filters.")
        st.stop()

    # ML predictions using lookup tables for every cuisine
    results = []
    for _, row in top_df.iterrows():
        pred, conf, _ = predict_demand(selected_location, row["cuisine"])
        price_bucket  = predict_price(selected_location, row["cuisine"])
        demand_label  = "✅ High Demand" if pred == 1 else "⚠️ Low Demand"
        conf_label    = f"{conf}% confident"
        results.append({
            "demand_pred":  pred,
            "demand_conf":  conf,
            "demand_text":  demand_label,
            "conf_label":   conf_label,
            "price_bucket": price_bucket
        })

    results_df = pd.DataFrame(results, index=top_df.index)
    top_df = pd.concat([top_df, results_df], axis=1)

    best       = top_df.iloc[0]
    best_color = CLUSTER_COLORS.get(best["cluster_name"], "#888")

    st.subheader("🎯 Top Recommendation")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🍽️ Cuisine",    best["cuisine"])
    c2.metric("⭐ Rating",      round(best["rating"], 2))
    c3.metric("⚔️ Competition", int(best["competition"]))
    c4.metric("💰 Price Tier",  best["price_bucket"])
    c5.metric("🤖 Confidence",  best["conf_label"])

    st.markdown(
        f"**Cluster:** <span style='background:{best_color}; color:white; padding:3px 10px; border-radius:12px'>{best['cluster_name']}</span>"
        f"&nbsp;&nbsp; **Demand Forecast:** {best['demand_text']}",
        unsafe_allow_html=True
    )
    st.success(f"Recommended: Launch a **{best['cuisine']}** kitchen in **{selected_location}** 🚀")
    st.divider()

    fig = px.bar(
        top_df, x="cuisine", y="custom_score", color="cluster_name",
        color_discrete_map=CLUSTER_COLORS, text="custom_score",
        title=f"Top {top_n} Cuisines in {selected_location}",
        labels={"custom_score": "Opportunity Score", "cuisine": "Cuisine", "cluster_name": "Cluster"}
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_tickangle=-30, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        top_df[["cuisine", "rating", "competition", "custom_score", "cluster_name", "demand_text", "conf_label", "price_bucket"]].rename(columns={
            "cuisine":      "Cuisine",
            "rating":       "Rating",
            "competition":  "Competition",
            "custom_score": "Opportunity Score",
            "cluster_name": "Cluster",
            "demand_text":  "ML Demand Forecast",
            "conf_label":   "Confidence",
            "price_bucket": "Price Tier"
        }),
        use_container_width=True, hide_index=True
    )

# ============================================================
# MODE 2 — Predict Any Combination
# ============================================================
else:

    st.subheader("🎯 Predict Any Location + Cuisine Combination")
    st.markdown(
        "Enter any location and cuisine — even combinations that don't currently exist. "
        "Both models use lookup table averages so any combination can be predicted."
    )

    col1, col2 = st.columns(2)
    with col1:
        sel_location = st.selectbox("📍 Select Location", sorted(df["location"].unique()))
    with col2:
        sel_cuisine  = st.selectbox("🍽️ Select Cuisine",  sorted(df["cuisine"].unique()))

    combo_exists = not df[(df["location"] == sel_location) & (df["cuisine"] == sel_cuisine)].empty
    if combo_exists:
        st.info("ℹ️ This combination exists in the dataset. Predicting from learned market patterns.")
    else:
        st.success("✨ New combination not in dataset — predicting from learned patterns!")

    if st.button("🔮 Predict", type="primary"):

        pred, conf, demand_inp = predict_demand(sel_location, sel_cuisine)
        price_bucket           = predict_price(sel_location, sel_cuisine)

        demand_label  = "✅ High Demand Expected" if pred == 1 else "⚠️ Low Demand Expected"
        verdict_color = "#2ecc71" if pred == 1 else "#e67e22"
        conf_label    = f"{conf}% confident this is {'High' if pred == 1 else 'Low'} Demand"

        st.divider()
        st.subheader("📊 Prediction Results")

        c1, c2, c3 = st.columns(3)
        c1.metric("📈 Demand Forecast",       "High" if pred == 1 else "Low")
        c2.metric("🤖 Confidence",             f"{conf}%")
        c3.metric("💰 Recommended Price Tier", price_bucket)

        st.markdown(
            f"<div style='background:{verdict_color}; color:white; padding:12px 20px; "
            f"border-radius:10px; font-size:18px; font-weight:bold; margin:10px 0'>"
            f"{demand_label} — {conf_label}</div>",
            unsafe_allow_html=True
        )

        # Show what the models used
        st.divider()
        st.subheader("🔎 What the Models Used")
        st.markdown("Both models derive their inputs from lookup table averages — not individual restaurant values:")

        d_comp  = mdl["demand_loc_comp"].get(sel_location,  mdl["demand_fallbacks"]["competition"])
        d_rat   = mdl["demand_cui_rat"].get(sel_cuisine,    mdl["demand_fallbacks"]["rating"])
        d_price = mdl["demand_cui_price"].get(sel_cuisine,  mdl["demand_fallbacks"]["price"])
        p_comp  = mdl["price_loc_comp"].get(sel_location,   mdl["price_fallbacks"]["competition"])
        p_dem   = mdl["price_loc_dem"].get(sel_location,    mdl["price_fallbacks"]["demand"])
        p_norm  = mdl["price_loc_norm"].get(sel_location,   mdl["price_fallbacks"]["normalized"])
        p_rat   = mdl["price_cui_rat"].get(sel_cuisine,     mdl["price_fallbacks"]["rating"])

        inp_df = pd.DataFrame({
            "Model":   [
                "Demand Predictor", "Demand Predictor", "Demand Predictor",
                "Price Recommender", "Price Recommender", "Price Recommender", "Price Recommender"
            ],
            "Feature": [
                "Avg Rating for Cuisine", "Avg Competition in Location", "Avg Price for Cuisine",
                "Avg Demand in Location", "Avg Rating for Cuisine",
                "Avg Competition in Location", "Avg Norm. Demand in Location"
            ],
            "Value": [
                round(d_rat, 2), round(d_comp, 2), round(d_price, 2),
                round(p_dem, 2), round(p_rat, 2), round(p_comp, 2), round(p_norm, 4)
            ],
            "Source": [
                "Cuisine average across all locations",
                "Location average across all cuisines",
                "Cuisine average across all locations",
                "Location average across all cuisines",
                "Cuisine average across all locations",
                "Location average across all cuisines",
                "Location average across all cuisines"
            ]
        })
        st.dataframe(inp_df, use_container_width=True, hide_index=True)

        # SHAP waterfall — live per prediction
        st.divider()
        st.subheader("🔍 Why this demand prediction?")
        st.markdown("Each bar shows how much that feature pushed the prediction toward High or Low Demand.")
        shap_vals = mdl["explainer"].shap_values(demand_inp)
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[0],
                base_values=mdl["explainer"].expected_value,
                data=demand_inp.iloc[0].values,
                feature_names=mdl["demand_features"]
            ),
            show=False
        )
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Price tier reference
        pr = mdl["price_ranges"]
        st.divider()
        st.subheader("💰 Price Tier Reference")
        c1, c2, c3 = st.columns(3)
        c1.info(f"🟢 Budget\n\n₹{int(pr['min']['Budget'])} – ₹{int(pr['max']['Budget'])}")
        c2.warning(f"🟡 Mid\n\n₹{int(pr['min']['Mid'])} – ₹{int(pr['max']['Mid'])}")
        c3.error(f"🔴 Premium\n\n₹{int(pr['min']['Premium'])} – ₹{int(pr['max']['Premium'])}")

# ------------------------------
# Global SHAP — always visible in expander
# ------------------------------
with st.expander("📊 View Global Feature Importance (SHAP)"):
    base_dir  = os.path.dirname(os.path.abspath(__file__))
    shap_path = os.path.join(base_dir, "..", "models", "plots", "shap_summary.png")
    imp_path  = os.path.join(base_dir, "..", "models", "plots", "shap_importance.png")
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SHAP Summary — how each feature influences demand predictions", use_container_width=True)
    if os.path.exists(imp_path):
        st.image(imp_path,  caption="SHAP Feature Importance — overall ranking", use_container_width=True)

# ------------------------------
# Cluster Summary — always visible
# ------------------------------
st.divider()
st.subheader("🗂️ Cluster Summary for Selected Location")
loc_for_summary = selected_location if mode == "🔍 Explore a Location" else sorted(df["location"].unique())[0]
cluster_summary = (
    df[df["location"] == loc_for_summary]
    .groupby("cluster_name")
    .agg(
        Cuisines=("cuisine", "count"),
        Avg_Demand=("demand", "mean"),
        Avg_Rating=("rating", "mean"),
        Avg_Competition=("competition", "mean")
    )
    .round(2).reset_index()
    .rename(columns={"cluster_name": "Market Cluster"})
)
st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

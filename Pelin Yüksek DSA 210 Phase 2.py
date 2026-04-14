#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 19:46:53 2026

@author: pelinyuksek
"""

# DSA 210 Term Project - Phase 2
# Şükran Pelin Yüksek DSA 210 Project Phase 2 
# Sephora Skincare Reviews: EDA & Hypothesis Tests

#Analyzing the Consistency Between Customer Ratings and Review Text
#Across Skincare Product Categories: A Study of Sephora Reviews

#Dataset: Sephora Products and Skincare Reviews (Kaggle)
#Dataset: 60,000 reviews from 6 main skincare subcategories

# I randomly split the dataset into 60000 because of the data being too big I tried to include all the important relevant information I needed to do these tests. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
os.makedirs("figures", exist_ok=True)

# Categories and price groups i'll use throughout the analysis
categories = ["Moisturizers", "Treatments", "Cleansers", "Eye Care", "Masks", "Sunscreen"]
price_groups = ["Budget", "Mid-range", "Luxury"]

# Loading the data
df = pd.read_csv("skincare_analysis_data.csv", low_memory=False)

# Making sure numeric columns are actually numeric
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["vader_compound"] = pd.to_numeric(df["vader_compound"], errors="coerce")
df["review_length"] = pd.to_numeric(df["review_length"], errors="coerce")
df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
df = df.dropna(subset=["rating", "vader_compound"])

print(f"Dataset loaded: {df.shape[0]} reviews, {df.shape[1]} columns")


# --- Dataset Overview ---

print("\nReviews per subcategory:")
print(df["secondary_category"].value_counts().to_string())

print("\nRating distribution:")
print(df["rating"].value_counts().sort_index().to_string())

print("\nPrice segment distribution:")
print(df["price_segment"].value_counts().reindex(price_groups).to_string())

print("\nSentiment distribution:")
print(df["sentiment_label"].value_counts().to_string())

print("\nMismatch distribution:")
print(df["mismatch_type"].value_counts().to_string())

print("\nDescriptive statistics:")
print(df[["rating", "vader_compound", "review_length", "price_usd"]].describe().round(4).to_string())


# --- EDA Visualizations ---

# fig 1 - rating distribution across subcategories (grouped bar)
rating_pcts = pd.crosstab(df["secondary_category"], df["rating"], normalize="index") * 100
rating_pcts = rating_pcts.reindex(categories)

fig, ax = plt.subplots(figsize=(10, 6))
rating_pcts.plot(kind="bar", ax=ax, colormap="RdYlGn")
ax.set_title("Rating Distribution by Subcategory")
ax.set_xlabel("Subcategory")
ax.set_ylabel("Percentage (%)")
ax.legend(title="Rating", bbox_to_anchor=(1.02, 1))
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("figures/fig1_rating_distribution.png")
plt.show()

# fig 2 - Sentiment Distribution per Subcategory (grouped bar)
sentiment_pcts = pd.crosstab(df["secondary_category"], df["sentiment_label"], normalize="index") * 100
sentiment_pcts = sentiment_pcts.reindex(categories)[["Positive", "Neutral", "Negative"]]

fig, ax = plt.subplots(figsize=(10, 6))
sentiment_pcts.plot(kind="bar", ax=ax, color=["#66c2a5", "#fee08b", "#fc8d62"])
ax.set_title("Sentiment Distribution by Subcategory")
ax.set_xlabel("Subcategory")
ax.set_ylabel("Percentage (%)")
ax.legend(title="Sentiment")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("figures/fig2_sentiment_distribution.png")
plt.show()

# fig 3 - mismatch rate per subcategory
df["is_mismatch"] = (df["mismatch_type"] != "Consistent").astype(int)
mismatch_rates = df.groupby("secondary_category")["is_mismatch"].mean().reindex(categories) * 100

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(mismatch_rates.index, mismatch_rates.values,
              color=sns.color_palette("Set2", len(categories)))
ax.set_title("Review Mismatch Rate by Subcategory")
ax.set_xlabel("Subcategory")
ax.set_ylabel("Mismatch Rate (%)")
for bar, val in zip(bars, mismatch_rates.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
            f"{val:.1f}%", ha="center", fontsize=10)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("figures/fig3_mismatch_rate.png")
plt.show()

# fig 4 - Average Rating by Price Segment
price_means = df.groupby("price_segment")["rating"].mean().reindex(price_groups)

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(price_means.index, price_means.values,
              color=["#66c2a5", "#fc8d62", "#8da0cb"])
ax.set_title("Average Rating by Price Segment")
ax.set_xlabel("Price Segment")
ax.set_ylabel("Average Rating")
ax.set_ylim(3.5, 5.0)
for bar, val in zip(bars, price_means.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("figures/fig4_price_rating.png")
plt.show()

# fig 5 - Review Length Distribution
fig, ax = plt.subplots(figsize=(10, 6))
for cat in categories:
    subset = df[df["secondary_category"] == cat]["review_length"]
    ax.hist(subset[subset <= 300], bins=50, alpha=0.5, label=cat)
ax.set_title("Review Length Distribution by Subcategory")
ax.set_xlabel("Word Count")
ax.set_ylabel("Frequency")
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig5_review_length.png")
plt.show()

# fig 6 - Average Sentiment Score by Star Rating
fig, ax = plt.subplots(figsize=(8, 6))
avg_sent = df.groupby("rating")["vader_compound"].mean()
colors_rating = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]
bars = ax.bar(avg_sent.index, avg_sent.values, color=colors_rating)
ax.set_title("Average Sentiment Score by Star Rating")
ax.set_xlabel("Star Rating")
ax.set_ylabel("Average VADER Score")
for bar, val in zip(bars, avg_sent.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{val:.3f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig("figures/fig6_rating_vs_sentiment.png")
plt.show()

# fig 7 - Mismatch Type Breakdown (stacked bar)
mismatch_crosstab = pd.crosstab(df["secondary_category"], df["mismatch_type"], normalize="index") * 100
mismatch_crosstab = mismatch_crosstab.reindex(categories)

fig, ax = plt.subplots(figsize=(10, 6))
mismatch_crosstab.plot(kind="bar", stacked=True, ax=ax,
                        color=["#66c2a5", "#fc8d62", "#8da0cb"])
ax.set_title("Mismatch Type Breakdown by Subcategory")
ax.set_xlabel("Subcategory")
ax.set_ylabel("Percentage (%)")
ax.legend(title="Mismatch Type", fontsize=9)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("figures/fig7_mismatch_breakdown.png")
plt.show()

# fig 8 - Correlation Heatmap
numeric_cols = ["rating", "vader_compound", "review_length", "price_usd", "is_mismatch"]
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm",
            center=0, square=True, ax=ax)
ax.set_title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("figures/fig8_correlation_heatmap.png")
plt.show()

print("All figures saved to figures/ folder")


# --- Hypothesis Tests ---

# test 1: Do ratings differ across subcategories (ANOVA)
print("\n--- Test 1: One-Way ANOVA (Ratings) ---")
print("H0: Mean ratings are equal across all subcategories")
print("H1: At least one subcategory has a different mean rating\n")

groups = [df[df["secondary_category"] == c]["rating"] for c in categories]
f_stat, p_val = stats.f_oneway(*groups)

for c in categories:
    print(f"  {c}: mean = {df[df['secondary_category'] == c]['rating'].mean():.4f}")
print(f"\nF = {f_stat:.4f}, p = {p_val:.2e}")
print(f"Result: {'Reject H0' if p_val < 0.05 else 'Fail to reject H0'} (alpha=0.05)")

# test 2: Moisturizers vs Treatments (t-test)
print("\n--- Test 2: T-Test (Moisturizers vs Treatments) ---")
print("H0: Mean rating of Moisturizers = Mean rating of Treatments")
print("H1: They are different\n")

moist = df[df["secondary_category"] == "Moisturizers"]["rating"]
treat = df[df["secondary_category"] == "Treatments"]["rating"]
t_stat, p_val2 = stats.ttest_ind(moist, treat)

print(f"Moisturizers: mean={moist.mean():.4f}, n={len(moist)}")
print(f"Treatments:   mean={treat.mean():.4f}, n={len(treat)}")
print(f"\nt = {t_stat:.4f}, p = {p_val2:.4f}")
print(f"Result: {'Reject H0' if p_val2 < 0.05 else 'Fail to reject H0'} (alpha=0.05)")

# test 3: Do sentiment scores differ across subcategories (ANOVA)
print("\n--- Test 3: One-Way ANOVA (Sentiment Scores) ---")
print("H0: Mean VADER scores are equal across all subcategories")
print("H1: At least one subcategory has a different mean score\n")

groups_sent = [df[df["secondary_category"] == c]["vader_compound"] for c in categories]
f_stat3, p_val3 = stats.f_oneway(*groups_sent)

for c in categories:
    print(f"  {c}: mean = {df[df['secondary_category'] == c]['vader_compound'].mean():.4f}")
print(f"\nF = {f_stat3:.4f}, p = {p_val3:.2e}")
print(f"Result: {'Reject H0' if p_val3 < 0.05 else 'Fail to reject H0'} (alpha=0.05)")

# test 4: Is mismatch related to subcategory (chi-square)
print("\n--- Test 4: Chi-Square Test ---")
print("H0: Mismatch occurrence is independent of subcategory")
print("H1: They are dependent\n")

contingency = pd.crosstab(df["secondary_category"], df["is_mismatch"])
chi2, p_val4, dof, expected = stats.chi2_contingency(contingency)

contingency.columns = ["Consistent", "Mismatch"]
print(contingency.to_string())
print(f"\nChi2 = {chi2:.4f}, df = {dof}, p = {p_val4:.4e}")
print(f"Result: {'Reject H0' if p_val4 < 0.05 else 'Fail to reject H0'} (alpha=0.05)")

# test 5: Budget vs Luxury ratings (t-test)
print("\n--- Test 5: T-Test (Budget vs Luxury) ---")
print("H0: Mean rating of Budget = Mean rating of Luxury")
print("H1: They are different\n")

budget = df[df["price_segment"] == "Budget"]["rating"]
luxury = df[df["price_segment"] == "Luxury"]["rating"]
t_stat5, p_val5 = stats.ttest_ind(budget, luxury)

print(f"Budget: mean={budget.mean():.4f}, n={len(budget)}")
print(f"Luxury: mean={luxury.mean():.4f}, n={len(luxury)}")
print(f"\nt = {t_stat5:.4f}, p = {p_val5:.2e}")
print(f"Result: {'Reject H0' if p_val5 < 0.05 else 'Fail to reject H0'} (alpha=0.05)")


# Summary

total_mismatch = df["is_mismatch"].sum()
mismatch_pct = df["is_mismatch"].mean() * 100
positive_pct = df[df["sentiment_label"] == "Positive"].shape[0] / len(df) * 100

print("\n\n--- Summary ---")
print(f"Total reviews analyzed: {len(df)}")
print(f"Positive reviews: {positive_pct:.1f}%")
print(f"Overall mismatch rate: {mismatch_pct:.1f}% ({total_mismatch} reviews)")
print(f"\nTest results:")
print(f"  ANOVA (ratings across categories): p={p_val:.2e} -> {'Significant' if p_val < 0.05 else 'Not significant'}")
print(f"  T-test (moisturizers vs treatments): p={p_val2:.4f} -> {'Significant' if p_val2 < 0.05 else 'Not significant'}")
print(f"  ANOVA (sentiment across categories): p={p_val3:.2e} -> {'Significant' if p_val3 < 0.05 else 'Not significant'}")
print(f"  Chi-square (mismatch vs category): p={p_val4:.4e} -> {'Significant' if p_val4 < 0.05 else 'Not significant'}")
print(f"  T-test (budget vs luxury): p={p_val5:.2e} -> {'Significant' if p_val5 < 0.05 else 'Not significant'}")

print("\nDone!")
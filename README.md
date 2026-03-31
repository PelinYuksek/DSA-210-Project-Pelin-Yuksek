# DSA-210-Project-Pelin-Yuksek
DSA 210 term project repository – Spring 2025-2026 
# DSA 210 Term Project

This repository contains my individual term project for DSA 210 Introduction to Data Science, Spring 2025-2026.

## Project Topic
Analyzing the Consistency Between Customer Ratings and Review Text Across Skincare Product Categories: A Study of Sephora Reviews 


## Project Description
This project investigates whether customer star ratings on Sephora skincare products are consistent with the sentiment expressed in their written reviews. The analysis compares skincare subcategories (moisturizers, cleansers, serums, masks, treatments) to understand where rating-review mismatches occur most frequently.


## Dataset

Primary Data: Sephora Products and Skincare Reviews— 8,000+ products, ~1M reviews
https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews 
Enrichment Data: Google Trends brand popularity scores for top skincare brands

## Planned Analysis

Exploratory Data Analysis: Category-level comparison of ratings, review lengths, and price-satisfaction patterns using visualizations (bar charts, box plots, heatmaps)
Hypothesis Testing: t-tests and ANOVA to test for statistically significant differences across categories
Sentiment-Rating Consistency Analysis: Using VADER to compute sentiment scores and detect mismatches between review text and star ratings
Machine Learning: Supervised and unsupervised methods to classify and cluster review patterns


## Enrichment Features

Sentiment score (VADER)
Review length (word count)
Sentiment-rating mismatch flag
Price segment (budget / mid-range / luxury)
Brand popularity index (Google Trends)

## Repository Structure
- `data/`: raw and processed datasets
- `notebooks/`: Jupyter notebooks
- `src/`: Python scripts
- `proposal/`: proposal document
- `reports/`: final report

## Author
Pelin Yüksek

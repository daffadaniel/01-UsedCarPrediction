# Used Car Price Prediction

This project focuses on building an end-to-end machine learning pipeline to predict used car prices using a large-scale Kaggle dataset (~188k rows).  
The emphasis is placed on **robust preprocessing, feature extraction from messy real-world data, and model validation**, rather than leaderboard optimization.

## Introduction

### Project Overview
Used car datasets often contain noisy categorical values, free-text engine descriptions, and inconsistent missing values.  
This project addresses these challenges by transforming unstructured automotive data into structured, model-ready features.

### Objectives
- Build a reproducible preprocessing pipeline for messy tabular data
- Extract meaningful numerical and categorical features from free-text engine descriptions
- Compare gradient boosting models on a realistic validation setup
- Demonstrate production-oriented ML practices

### Data Source
Kaggle Playground Series S4E9 — Used Car Price Dataset  
~188,000 listings with attributes such as brand, mileage, engine details, fuel type, and transmission.

## Methodology

### Data Preparation
Key preprocessing decisions include:
- Standardizing inconsistent categorical values
- Parsing engine descriptions into structured features (horsepower, cylinders, electric flag)
- Context-aware imputation for electric vs non-electric vehicles
- Cardinality reduction for color and model attributes

### Modeling Strategy
Multiple gradient boosting models (LightGBM, XGBoost, CatBoost) were evaluated using cross-validation and a hold-out validation set.  
Model selection prioritized **generalization performance and training stability**.

## Results & Findings
- LightGBM achieved the best validation RMSE (68,090)
- Engine-derived features significantly improved predictive performance
- Proper preprocessing had a larger impact than model choice

## Key Takeaways
- Feature engineering from unstructured text can outperform model complexity
- Leakage-safe pipelines are critical in real-world ML systems
- Gradient boosting models excel when supported by strong data preparation

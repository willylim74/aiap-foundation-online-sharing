# Predicting News Article Shares

The objective is to predict the number of shares (popularity) on social media platforms for the news articles using the dataset provided.

We will evaluate at least 3 suitable models for predicting the number of shares.

# 
# Prerequisites and Installation

Refer to **'requirements.txt'** file for prerequisites needed to run the project.

# 
# Instructions for Executing the Pipeline
## main.py

Execute **'main.py'** directly from repo folder.
In the **'config.yaml'** file, we can modify 'hyperparameters' to set various hyperparameters for tuning of various models.

# 
# Description of Logical Steps/Flow of the Pipeline
## data_preparation.py

In **'data_preparation.py'**, we perform the logical steps and flow of our machine learning pipeline
- data cleaning
- feature engineering
- preprocessing (pipeline for each model)

#
## Data Cleaning
- 35,680 articles, 29 raw features + target `shares`.

- Missing values: `num_videos` (~47%, imputed as 0 — no video embedded), `data_channel` (~15%, imputed as `"unknown"` category). 

# 
## Pre-Processing
The preprocessing pipeline consists of:

1. Imputer
`num_hrefs`, `num_self_hrefs`, `num_imgs`, and `self_reference_*` (~2% missing) are median-imputed **inside the preprocessing pipeline, fit on the training split only** 

Note: imputing with a full-dataset median before the split would leak test-set information into training.

2. Feature Engineering
Feature engineering is performed **inside the preprocessing pipeline, fit on the training split only** to prevent data leakage.

Engineered features reflect metrics online publishers/social editors track when optimizing for shares. 
- `title_length_band`: Headlines of ~6–12 words are known to perform best on social feeds 
- `lexical_density`: Content richness readability proxy
- `imgs_per_1k_words`, `videos_per_1k_words`, `media_richness`: Visual content increases scannability and shareability
- `self_href_ratio`: Internal-linking (SEO) ratio
- `keyword_spread_avg`, `keyword_quality_score`: Keyword-targeting strategy and performance 
- `self_reference_range`: "Network effect", articles linking to previously popular content tend to inherit traffic
- `is_weekend`: Weekend vs weekday posting/reading behavior
- `comments_per_1k_words`: Engagement density (length-normalized)
- `freshness_band`: Content age / discovery decay at time of data collection

3. Feature Scaling
- for Numerical features 
- Standardization (mean 0, std dev 1)

4. Feature Encoding 
- for Categorical features
- One-hot: for Nominal (Non-ordered) features
- Ordinal: there are no ordered features

#
# Model Development
## model_training.py
#
## Data Splitting 
- `shares` is heavily right-skewed (skew ≈ 34), typical of viral content metrics — most articles get modest engagement, a few go viral. 
- Target modeled as `log1p(shares)` to stabilize variance; metrics reported on both scales.

- **80/10/10 train-validata-test split**
- Training dataset: building the model
- Validation dataset: tuning and selection
- Test dataset: Final evaluation

- predictions are performed on `log1p(shares)` and then inverted (`expm1`, clipped at 0) for share-scale metrics.

#
## Train and evaluate Baseline Models
Linear Models: 
**Linear Regression**
**Ridge**
**Lasso**

Non-linear models:
**Random Forest**
**Gradient Boosting** 
**XGBoost**

Evaluation Metrics:
**Mean Absolute Error**
**Root Mean Square Error**
**r2**

#
## Hyperparameter Tuning
We perform Random Search CV over a specified parameter grid in **'config.yaml'**, testing n_iter possible combination of hyperparameter values and evaluating each combination using 3-fold cross-validation to achieve better performance.

n_iter: 10
cv: 3
scoring: r2


## Final Model Evaluation
Evaluate the final model (best model from hyperparameter tuning) on the Test dataset

# 
# Key Findings from EDA
## eda.jpynb
#
1. Data Overview
- 35,680 articles, 29 raw features + target `shares`.
`shares` is heavily right-skewed, typical of viral content metrics

- Missing values: `num_videos` (~47%), `data_channel` (~15%), and a handful of link/image/self-reference columns (~2%).

- No duplicates. 

2. Target variable
- Target variable `shares` is heavily right-skewed (skew ≈ 34), typical of viral content metrics
most articles get modest engagement, a few go viral. 

- A log transformation 'log_shares' is applied to stabilize variance and improve model performance.

3. Numerical Columns
- 27 numerical columns, including content features (text stats) and engagement features (comments, references)

- Skewed distributions are observed in many numerical features, which may require transformations or scaling for modeling.

- num_videos: ~47% missing. a missing video count almost always means "no video embedded" (videos are optional media), so we impute with '0' rather than the mean, which would understate articles that truly have no video.

- num_hrefs, num_self_hrefs, num_imgs, self_reference_*, a small fraction (~2%) missing, we impute with the median (robust to the heavy right-skew of these count variables)

- n_comments has strong positive correlation with shares, indicating that articles with more comments tend to have higher shares. However, n_comments is not available at the time of publishing of the article, so it cannot be used as a predictive feature.

- Remaining features show weak/moderate correlation with log_shares, suggesting limited linear relationships.

4. Categorical Columns
- categorical columns: weekday, data_channel
- Weekend Effect: Saturday has the highest shares (~4158). Sunday is also relatively high. Users are more active on weekends
- data_channel: missing means the article was not tagged with a channel, treat as its own category "unknown" rather than dropping ~15% of rows

#
# Explanation of Model Choices
This is a regression problem on a highly-skewed target. 

We select both linear and non-linear regression models, with expectation that non-linear regression models should perform better.

Linear models:
1. Linear Regression
2. Ridge
3. Lasso

Non-linear models:
4. Random Forest
5. Gradient Boosting
6. XGBoost


# 
# Evaluation of Models
We evaluate the models using the following performance metrics for regression problems:

1. Mean Absolute Error
2. Root Mean Square Error
3. r2

Predictions are performed on `log1p(shares)` (log-transformed target) and then inverted (`expm1`, clipped at 0) for share-scale metrics. Models are ranked based on R² on log scale.


**Gradient Boosting** is the best performer, nearly doubling the linear models' baselines' R² on the log scale by capturing non-linear relationships and feature interactions. All linear models have very low R² (less than 0.08) on the raw `shares` scale — they're dragged down by a handful of extreme viral outliers even after log-modeling and clipping, which the tree ensembles handle more gracefully.

# 

# Considerations for Deployment
The non-linear models are computationally intensive, especially if using Grid Search CV to tune hyperparameters. We have chosen Random Search and limited the number of iterations to 10, and cross-validation to 3 for efficiency.



# Standard library imports
import logging
from typing import Any, Dict
import pandas as pd
import numpy as np

# Related third-party imports
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer for feature engineering.
    This ensures all transformations are applied consistently within the pipeline.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # =========================
        # Feature Engineering
        # =========================
        

        # (a) Headline length -- industry rule of thumb: titles of ~6-12 words
        #     tend to perform best on social feeds; flag short/long/optimal

        df["title_length_band"] = pd.cut(
            df["n_tokens_title"],
            bins=[0, 6, 12, np.inf],
            labels=["short", "optimal", "long"]
        )

        # (b) Content depth: total content length already exists (n_tokens_content);
        #     add a readability-adjacent measure (avg words per "idea" proxied by
        #     the ratio of unique tokens to total content length -> lexical density)

        df["lexical_density"] = df["n_unique_tokens"] * df["n_non_stop_words"]

        # (c) Visual richness: images/videos per 1000 words of content - editors use
        #     this to gauge how "scannable"/shareable an article is on social feeds

        # helper column to avoid division by zero; will be dropped after feature engineering
        df["content_length_safe"] = df["n_tokens_content"].replace(0, np.nan)

        df["imgs_per_1k_words"] = (df["num_imgs"] / df["content_length_safe"] * 1000).fillna(0)
        df["videos_per_1k_words"] = (df["num_videos"] / df["content_length_safe"] * 1000).fillna(0)
        df["media_richness"] = df["imgs_per_1k_words"] + df["videos_per_1k_words"]

        # (d) Internal linking ratio (SEO / on-site engagement metric): 
        #     what fraction of an article's links point back to the same site

        df["self_href_ratio"] = np.where(
            df["num_hrefs"] > 0,
            df["num_self_hrefs"] / df["num_hrefs"],
            0
        )

        # (e) Keyword performance spread: gap between the best- and worst-performing
        #     keyword's average shares - large gap suggests reliance on a single
        #     "hit" keyword; publishers use this to judge keyword-strategy risk

        df["keyword_spread_avg"] = df["kw_max_avg"] - df["kw_min_avg"]
        df["keyword_quality_score"] = df["kw_avg_avg"]

        # (f) Evergreen / network-effect signal: how well articles this article
        #     links to have historically performed - a strong predictor since
        #     articles referencing popular content tend to inherit some traffic

        df["self_reference_range"] = (
            df["self_reference_max_shares"] - df["self_reference_min_shares"]
        )

        # (g) Publish-day effects: weekend posts often see different sharing
        #     behaviour (more leisure browsing) than weekday posts

        df["is_weekend"] = df["weekday"].isin(["saturday", "sunday"]).astype(int)

        # (h) Reader engagement proxy: comments per 1000 words (engagement density
        #     rather than raw comment count, which is length-biased)

        df["comments_per_1k_words"] = (
            df["n_comments"] / df["content_length_safe"] * 1000
        ).fillna(0)

        # (i) Content freshness bucket: very recent vs older articles at time of
        #     collection - proxies algorithmic/organic discovery decay

        df["freshness_band"] = pd.cut(
            df["timedelta"],
            bins=[-1, 90, 365, np.inf],
            labels=["recent", "moderate", "old"]
        )
        # drop helper column no longer needed
        df = df.drop(columns=["content_length_safe"])
        

        return df
    

class DataPreparation:
    """
    Handles data cleaning, missing value handling, and preprocessing pipeline construction.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = self._create_preprocessor()

    # =====================================================
    # STATIC METHODS FOR MISSING VALUE HANDLING
    # =====================================================

    @staticmethod
    def _fill_categorical(df: pd.DataFrame) -> pd.DataFrame:
        """Fill categorical missing values."""
        df["data_channel"] = df["data_channel"].fillna("unknown")
        return df

    @staticmethod
    def _fill_zero(df: pd.DataFrame) -> pd.DataFrame:
        """Fill count-based columns with zero."""
        df["num_videos"] = df["num_videos"].fillna(0)
        return df



    # =====================================================
    # CLEANING FUNCTION
    # =====================================================

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies basic cleaning and missing value handling.
        Feature engineering is NOT done here.
        """
        logging.info("Starting data cleaning.")

        df = df.drop(columns=["ID", "URL"])

        # Apply static missing value handlers
        df = self._fill_categorical(df)
        df = self._fill_zero(df)

        logging.info(f"Remaining missing values:\n\n{df.isnull().sum()[df.isnull().sum() > 0]}")
        logging.info("Data cleaning completed.")
        return df


    # =====================================================
    # PREPROCESSOR (INCLUDES FEATURE ENGINEERING)
    # =====================================================

    def _create_preprocessor(self) -> Pipeline:
        """
        Builds a full preprocessing pipeline including:
        - Feature engineering
        - Scaling
        - Encoding
        """
        # Numerical Transformer - for numerical features
        # median-impute
        # standardise (needed for the linear/regularised models; harmless for tree ensembles).
        
        numerical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Nominal Transformer - for categorical variables with no order
        # most_frequent-impute (safety net)
        # One-hot encoding
        
        nominal_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        # Ordinal Transformer - for categorical variables with a known order
        # There is no ordinal encoding in the current dataset, but this is a placeholder for future use.
        # ordinal_transformer = Pipeline(steps=[
        #     ("imputer", SimpleImputer(strategy="most_frequent")),
        #     ("ordinal", OrdinalEncoder(
        #         categories=[self.config["ordinal_features"]],
        #         handle_unknown="use_encoded_value",
        #         unknown_value=-1
        #     ))
        # ])

        column_transformer = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, self.config["numerical_features"]),
                ("nom", nominal_transformer, self.config["nominal_features"]),
            #    ("ord", ordinal_transformer, self.config["ordinal_features"]),
                ("pass", "passthrough", self.config["passthrough_features"]),
            ],
            remainder="passthrough",
            n_jobs=-1
        )

        # Preprocessor: Feature Engineering + ColumnTransformer
        preprocessor = Pipeline([
            ("feature_engineering", FeatureEngineer()),
            ("preprocessing", column_transformer)
        ])

        return preprocessor






    

    
# lab2_utils.py — shared data preparation for Lab 2


ENGINEERED_FEATURES = [
    "HouseAge",
    "YearsSinceRemodel",
    "WasRemodeled",
    "TotalArea",
    "TotalBathrooms"
]

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PRICE_BINS = [0, 100000, 130000, 180000, 250000, 400000, np.inf]
CLASS_NAMES = ["<100k", "100–130k", "130–180k", "180–250k", "250–400k", "≥400k"]


def load_data(path="data/AmesHousing.csv"):
    """Read the dataset; 'NA' stays a category (= the feature does not exist)."""
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])


def clean_data(df):
    """Corrections found in Lab 1."""
    df = df.copy()
    # typo: garage built in 2207 -> 2007
    df.loc[df["Garage Yr Blt"] == 2207, "Garage Yr Blt"] = 2007
    # missing basement values = house without a basement
    bsmt_cols = ["BsmtFin SF 1", "BsmtFin SF 2", "Bsmt Unf SF",
                 "Total Bsmt SF", "Bsmt Full Bath", "Bsmt Half Bath"]
    df[bsmt_cols] = df[bsmt_cols].fillna(0)
    return df


def make_target(df):
    """The 6 price classes from the assignment, labels 0-5."""
    return pd.cut(df["SalePrice"], bins=PRICE_BINS, labels=False, right=False).astype(int).rename("PriceClass")

def encode_features(df):
    """Baseline encoding: one-hot for all categorical features."""
    X = df.drop(columns=["Order", "PID", "SalePrice"])
    X["MS SubClass"] = X["MS SubClass"].astype(str)  # numbers, but categories
    cat_cols = X.select_dtypes(exclude="number").columns
    return pd.get_dummies(X, columns=cat_cols, dtype=int)


def get_data(path="data/AmesHousing.csv", engineered_features=None):
    """X (features) and y (classes 0-5), ready for XGBoost."""
    df = clean_data(load_data(path))

    if engineered_features:
        df = add_engineered_features(df, engineered_features)

    return encode_features(df), make_target(df)

def get_split(X, y, test_size=0.2):
    """The fixed train/test split everyone uses (stratified, random_state=42)."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)


def add_engineered_features(df, selected_features=None):
    """Add selected engineered features from Lab 1."""
    df = df.copy()

    if selected_features is None:
        selected_features = ENGINEERED_FEATURES

    unknown = set(selected_features) - set(ENGINEERED_FEATURES)
    if unknown:
        raise ValueError(f"Unknown engineered features: {unknown}")

    # Correct the 1950 artefact for houses built before 1950
    artefact = (
        (df["Year Built"] < 1950) &
        (df["Year Remod/Add"] == 1950)
    )

    engineered = {
        "HouseAge": (
            df["Yr Sold"] - df["Year Built"]
        ).clip(lower=0),

        "YearsSinceRemodel": (
            df["Yr Sold"] - df["Year Remod/Add"]
        ).clip(lower=0),

        "WasRemodeled": (
            (df["Year Remod/Add"] > df["Year Built"]) &
            ~artefact
        ).astype(int),

        "TotalArea": (
            df["Gr Liv Area"] + df["Total Bsmt SF"]
        ),

        "TotalBathrooms": (
            df["Full Bath"]
            + df["Bsmt Full Bath"]
            + 0.5 * (df["Half Bath"] + df["Bsmt Half Bath"])
        )
    }

    for feature in selected_features:
        df[feature] = engineered[feature]

    return df
# lab2_utils.py — shared data preparation for Lab 2
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


def get_data(path="data/AmesHousing.csv"):
    """X (features) and y (classes 0-5), ready for XGBoost."""
    df = clean_data(load_data(path))
    return encode_features(df), make_target(df)


def get_split(X, y, test_size=0.2):
    """The fixed train/test split everyone uses (stratified, random_state=42)."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
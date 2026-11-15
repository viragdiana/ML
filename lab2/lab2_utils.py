# lab2_utils.py — shared data preparation for Lab 2

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

PRICE_BINS = [0, 100000, 130000, 180000, 250000, 400000, np.inf]
CLASS_NAMES = ["<100k", "100–130k", "130–180k", "180–250k", "250–400k", "≥400k"]

ENGINEERED_FEATURES = [
    "HouseAge",
    "YearsSinceRemodel",
    "WasRemodeled",
    "TotalArea",
    "TotalBathrooms"
]


# Baseline data preparation (Part A)

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


# Engineered features from Lab 1 (Part B)

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


# Final data preparation (result of Part B) — used by Part C and the final evaluation

TOP3_FEATURES = ["TotalArea", "HouseAge", "TotalBathrooms"]

_QUALITY = {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_MAPPINGS = {
    "Exter Qual": _QUALITY, "Exter Cond": _QUALITY, "Bsmt Qual": _QUALITY, "Bsmt Cond": _QUALITY,
    "Heating QC": _QUALITY, "Kitchen Qual": _QUALITY, "Fireplace Qu": _QUALITY,
    "Garage Qual": _QUALITY, "Garage Cond": _QUALITY, "Pool QC": _QUALITY,
    "Bsmt Exposure": {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFin Type 1": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFin Type 2": {"NA": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "Lot Shape": {"IR3": 0, "IR2": 1, "IR1": 2, "Reg": 3},
    "Utilities": {"ELO": 0, "NoSeWa": 1, "NoSewr": 2, "AllPub": 3},
    "Land Slope": {"Sev": 0, "Mod": 1, "Gtl": 2},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
    "Garage Finish": {"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Paved Drive": {"N": 0, "P": 1, "Y": 2},
    "Fence": {"NA": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4},
}


def ordinal_encode(df):
    """Ordered categories -> numbers (the remaining categories are one-hot encoded later)."""
    df = df.copy()
    for col, mapping in ORDINAL_MAPPINGS.items():
        unknown = set(df[col].dropna().unique()) - set(mapping)
        if unknown:
            raise ValueError(f"Unexpected categories in {col}: {unknown}")
        df[col] = df[col].map(mapping)
    return df


def get_final_data(path="data/AmesHousing.csv"):
    """Final data preparation of Part B: top 3 features + ordinal encoding (253 features)."""
    df = clean_data(load_data(path))
    df = add_engineered_features(df, TOP3_FEATURES)
    return encode_features(ordinal_encode(df)), make_target(df)


def balanced_weights(y):
    """Balanced sample weights: rare price classes get a higher weight during training."""
    return compute_sample_weight("balanced", y)


def cv_macro_f1(model, X, y, cv=None, weights=True):
    """Macro-F1 per fold; balanced weights are computed on the training part of each fold."""
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold_train, fold_val in cv.split(X, y):
        X_a, y_a = X.iloc[fold_train], y.iloc[fold_train]
        m = clone(model)
        m.fit(X_a, y_a, sample_weight=balanced_weights(y_a) if weights else None)
        scores.append(f1_score(y.iloc[fold_val], m.predict(X.iloc[fold_val]), average="macro"))
    return np.array(scores)
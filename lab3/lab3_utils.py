# lab2_utils.py — shared data preparation for Lab 2 updated for LAb 3
# The changes are fill_missing (new), encode_features (new ordinal parameter) and get_data. +
#  Lab 3: PCA and k-means (shortcuts for the team; the notebook shows every step) it's at the bottom. 
# Ai grija la case noi Stefi!!! pentru part 3 de aia am facut project, sa ai piplenul
# si vedeti ca acum ca sunt foldere trebuie sa dati pathul cand faceti:
#  X, y = get_data(path="../data/AmesHousing.csv", impute=True, ordinal=True)
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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


def fill_missing(df):
    """Fill the remaining numerical NaN values (needed for PCA, XGBoost can handle them natively)."""
    df = df.copy()
    df["Lot Frontage"] = df["Lot Frontage"].fillna(
        df.groupby("Neighborhood")["Lot Frontage"].transform("median")
    )
    df["Lot Frontage"] = df["Lot Frontage"].fillna(df["Lot Frontage"].median())
    df["Mas Vnr Area"] = df["Mas Vnr Area"].fillna(0)
    df[["Garage Cars", "Garage Area"]] = df[["Garage Cars", "Garage Area"]].fillna(0)
    df["Garage Yr Blt"] = df["Garage Yr Blt"].fillna(df["Year Built"])
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    return df


def make_target(df):
    """The 6 price classes from the assignment, labels 0-5."""
    return pd.cut(df["SalePrice"], bins=PRICE_BINS, labels=False, right=False).astype(int).rename("PriceClass")


def encode_features(df, ordinal=False):
    """ordinal=True: ordered categories are mapped to numbers (ORDINAL_MAPPINGS), all remaining
    categorical columns (nominal ones) are one-hot encoded. ordinal=False: one-hot for all categorical features."""
    X = df.drop(columns=["Order", "PID", "SalePrice"])
    if ordinal:
        X = ordinal_encode(X)
    X["MS SubClass"] = X["MS SubClass"].astype(str)  # numbers, but categories
    cat_cols = X.select_dtypes(exclude="number").columns
    return pd.get_dummies(X, columns=cat_cols, dtype=int)


def get_data(path="data/AmesHousing.csv", engineered_features=None, impute=False, ordinal=False):
    """X (features) and y (classes 0-5). ordinal=True: ordinal encoding for ordered categories, one-hot for the rest.
    impute=True: fill the remaining numerical NaN values (needed for PCA)."""
    df = clean_data(load_data(path))

    if impute:
        df = fill_missing(df)

    if engineered_features:
        df = add_engineered_features(df, engineered_features)

    X = encode_features(df, ordinal=ordinal)

    if impute:
        X = X.fillna(X.median())

    return X, make_target(df)


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

# Lab 3: PCA and k-means (shortcuts for the team; the notebook shows every step)

VARIANCE_THRESHOLDS = (0.98, 0.95, 0.90, 0.80)

BEST_SCALING = "minmax"
BEST_THRESHOLD = 0.98
BEST_LOG_COLUMNS = None


def scale_data(X, method="minmax"):
    """Min-max ('minmax') or standardization ('standard'). Returns the scaled DataFrame and the fitted scaler."""
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    return X_scaled, scaler


def skewed_columns(X, threshold=1.0):
    """Numerical columns with |skewness| > threshold (binary and ordinal columns are excluded): candidates for np.log1p."""
    keep = (X.skew().abs() > threshold) & (X.nunique() > 2) & ~X.columns.isin(list(ORDINAL_MAPPINGS))
    return X.columns[keep.to_numpy()].tolist()


def log_transform(X, columns):
    """np.log1p on the given columns (apply before scaling)."""
    X = X.copy()
    X[columns] = np.log1p(X[columns])
    return X


def get_pca_data(scaling="minmax", thresholds=VARIANCE_THRESHOLDS, log_columns=None, path="data/AmesHousing.csv"):
    """Everything from part 1 in one call (part 2: all four thresholds). Returns a dict with:
    X (encoded features), y (price classes), X_scaled, scaler,
    pca (fitted PCA per threshold), Z (reduced data per threshold), log_columns."""
    X, y = get_data(path=path, impute=True, ordinal=True)
    if log_columns:
        X = log_transform(X, log_columns)
    X_scaled, scaler = scale_data(X, scaling)
    pcas, Z = {}, {}
    for t in thresholds:
        pcas[t] = PCA(n_components=t, svd_solver="full", random_state=42).fit(X_scaled)
        Z[t] = pcas[t].transform(X_scaled)
    return {"X": X, "y": y, "X_scaled": X_scaled, "scaler": scaler,
            "pca": pcas, "Z": Z, "log_columns": log_columns or []}


def get_all_variants(path="data/AmesHousing.csv"):
    """The four variants for part 2: min-max / standardization, each with and without log transform."""
    base = get_pca_data("minmax", path=path)
    log_cols = skewed_columns(base["X"])
    return {
        "minmax": base,
        "standard": get_pca_data("standard", path=path),
        "minmax_log": get_pca_data("minmax", log_columns=log_cols, path=path),
        "standard_log": get_pca_data("standard", log_columns=log_cols, path=path),
    }


def get_best_pca_data(path="data/AmesHousing.csv"):
    """The one final configuration (BEST_* constants) for part 3 and the bonuses. Returns a dict with:
    X, y, X_scaled, scaler, pca (one fitted PCA), Z (reduced data), threshold, log_columns."""
    d = get_pca_data(BEST_SCALING, thresholds=(BEST_THRESHOLD,), log_columns=BEST_LOG_COLUMNS, path=path)
    return {"X": d["X"], "y": d["y"], "X_scaled": d["X_scaled"], "scaler": d["scaler"],
            "pca": d["pca"][BEST_THRESHOLD], "Z": d["Z"][BEST_THRESHOLD],
            "threshold": BEST_THRESHOLD, "log_columns": d["log_columns"]}


def project(X_new, data, threshold=None):
    """Project new houses (encoded like X) into the PCA space. Missing one-hot columns are filled with 0,
    missing numerical columns with the median. Works with the result of get_best_pca_data (threshold=None)
    and of get_pca_data (threshold=0.98 if None)."""
    X_ref = data["X"]
    missing = [c for c in X_ref.columns if c not in X_new.columns]
    X_new = X_new.reindex(columns=X_ref.columns)
    for c in missing:
        X_new[c] = 0 if X_ref[c].nunique() <= 2 else X_ref[c].median()
    if data["log_columns"]:
        X_new = log_transform(X_new, data["log_columns"])
    pca = data["pca"]
    if isinstance(pca, dict):
        pca = pca[VARIANCE_THRESHOLDS[0] if threshold is None else threshold]
    return pca.transform(data["scaler"].transform(X_new))


def get_loadings(pca, feature_names):
    """Loadings = pca.components_.T * np.sqrt(pca.explained_variance_), one column per principal component."""
    return pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=feature_names,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )
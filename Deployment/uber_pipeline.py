"""
Shared preprocessing for Uber fare prediction (Task 2 pipeline).

Both the training notebook and the Flask app import this module, so raw trip
details are turned into model features by exactly the same code in both places.
The fitted model is saved as a full scikit-learn Pipeline, which unpickles only
if this module is importable -- keep it next to app.py.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# (latitude, longitude) of the reference points behind the *_dist features.
LANDMARKS = {
    "jfk": (40.6413, -73.7781),   # JFK Airport
    "ewr": (40.6895, -74.1745),   # Newark Airport
    "lga": (40.7769, -73.8740),   # LaGuardia Airport
    "sol": (40.6892, -74.0445),   # Statue of Liberty
    "nyc": (40.7128, -74.0060),   # NYC centre
}

# Columns a user (or the raw CSV) must provide.
RAW_INPUT_COLUMNS = [
    "pickup_datetime", "pickup_latitude", "pickup_longitude",
    "dropoff_latitude", "dropoff_longitude", "passenger_count",
    "Car Condition", "Weather", "Traffic Condition",
]

CAR_CONDITION_ORDER = ["Bad", "Good", "Very Good", "Excellent"]

NUMERIC_COLS = ["distance", "jfk_dist", "ewr_dist", "lga_dist", "sol_dist",
                "nyc_dist", "passenger_count", "bearing"]
ORDINAL_COLS = ["Car Condition"]
ONEHOT_COLS = ["Weather", "Traffic Condition"]
PASSTHROUGH_COLS = ["hour_sin", "hour_cos", "is_weekend", "day", "month", "year"]
FEATURE_COLUMNS = NUMERIC_COLS + ORDINAL_COLS + ONEHOT_COLS + PASSTHROUGH_COLS


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres."""
    lat1, lon1, lat2, lon2 = (np.radians(v) for v in (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2, convention="signed"):
    """Initial bearing in degrees: 'signed' = -180..180, 'compass' = 0..360."""
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    b = np.degrees(np.arctan2(x, y))
    return b % 360 if convention == "compass" else b


# ---------------------------------------------------------------------------
# Deterministic feature engineering (raw trip details -> model features)
# ---------------------------------------------------------------------------
def add_features(raw, bearing_convention="signed"):
    """Turn raw trip columns into the exact feature table the model expects."""
    d = pd.DataFrame(raw).copy()
    dt = pd.to_datetime(d["pickup_datetime"])
    plat = d["pickup_latitude"].astype(float)
    plon = d["pickup_longitude"].astype(float)
    dlat = d["dropoff_latitude"].astype(float)
    dlon = d["dropoff_longitude"].astype(float)

    out = pd.DataFrame(index=d.index)
    out["distance"] = haversine_km(plat, plon, dlat, dlon)
    for key, (lat, lon) in LANDMARKS.items():
        out[f"{key}_dist"] = haversine_km(plat, plon, lat, lon)
    out["passenger_count"] = d["passenger_count"].astype(float)
    out["bearing"] = bearing_deg(plat, plon, dlat, dlon, bearing_convention)

    out["Car Condition"] = d["Car Condition"]
    out["Weather"] = d["Weather"]
    out["Traffic Condition"] = d["Traffic Condition"]

    # Cyclical hour: hour 23 and hour 0 are neighbours, not 23 apart.
    out["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
    out["is_weekend"] = (dt.dt.weekday >= 5).astype(int)
    out["day"] = dt.dt.day
    out["month"] = dt.dt.month
    out["year"] = dt.dt.year
    return out[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Statistic-learning steps (fit on train only, inside the sklearn Pipeline)
# ---------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Caps each numeric column at percentiles learned on fit()."""

    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_ = X.quantile(self.lower_q)
        self.upper_ = X.quantile(self.upper_q)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.clip(lower=self.lower_, upper=self.upper_, axis=1).values


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("winsorize", Winsorizer(0.01, 0.99)),
        ("scale", RobustScaler()),
    ])
    ordinal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[CAR_CONDITION_ORDER])),
    ])
    onehot = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_COLS),
        ("ord", ordinal, ORDINAL_COLS),
        ("ohe", onehot, ONEHOT_COLS),
        ("pass", "passthrough", PASSTHROUGH_COLS),
    ])


def build_pipeline(regressor):
    """preprocessor -> regressor, with log1p/expm1 on the target (fare is right-skewed)."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", TransformedTargetRegressor(
            regressor=regressor, func=np.log1p, inverse_func=np.expm1)),
    ])

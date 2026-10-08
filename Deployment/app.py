"""
Uber Fare Predictor -- Flask app (Task 3).

Raw form values -> validation -> uber_pipeline.add_features (same feature
engineering as training) -> saved sklearn Pipeline (winsorize / encode / scale /
model) -> predicted fare.

The model and metadata are loaded ONCE when the app starts.
"""
import json
import math
import os
import sys
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

import uber_pipeline as up  # must be importable for joblib to unpickle the model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "uber_fare_model.pkl")
META_PATH = os.path.join(BASE_DIR, "model_meta.json")

# ---------------------------------------------------------------------------
# Load the trained pipeline + metadata once at startup
# ---------------------------------------------------------------------------
if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
    sys.exit(
        "Model files not found. Run every cell of train_and_compare_models.ipynb first -- "
        "it creates uber_fare_model.pkl and model_meta.json next to app.py."
    )

MODEL = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    META = json.load(f)

BEARING_CONVENTION = META["bearing_convention"]
CATEGORIES = META["categories"]
YEAR_MIN, YEAR_MAX = META["year_range"]

# Sensible input limits (the model only knows about NYC trips).
LAT_RANGE = (40.40, 41.10)
LON_RANGE = (-74.50, -73.40)
PASSENGER_RANGE = (1, 6)
MIN_TRIP_KM, MAX_TRIP_KM = 0.05, 100.0

LANDMARK_PRESETS = [
    ("Times Square", 40.7580, -73.9855),
    ("Central Park", 40.7829, -73.9654),
    ("Brooklyn Bridge", 40.7061, -73.9969),
    ("JFK Airport", 40.6413, -73.7781),
    ("LaGuardia Airport", 40.7769, -73.8740),
    ("Newark Airport", 40.6895, -74.1745),
    ("Statue of Liberty", 40.6892, -74.0445),
]

DEFAULT_VALUES = {
    "pickup_latitude": "40.7580", "pickup_longitude": "-73.9855",
    "dropoff_latitude": "40.6413", "dropoff_longitude": "-73.7781",
    "passenger_count": "1",
    "pickup_datetime": f"{YEAR_MAX}-06-15T18:30",
    "weather": "sunny" if "sunny" in CATEGORIES["Weather"] else CATEGORIES["Weather"][0],
    "traffic": "Flow Traffic" if "Flow Traffic" in CATEGORIES["Traffic Condition"] else CATEGORIES["Traffic Condition"][0],
    "car_condition": "Good",
}

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _parse_float(raw, label, lo, hi, errors, key):
    raw = (raw or "").strip()
    if raw == "":
        errors[key] = f"{label} is required."
        return None
    try:
        value = float(raw)
    except ValueError:
        errors[key] = f"{label} must be a number (got '{raw}')."
        return None
    if not math.isfinite(value):
        errors[key] = f"{label} must be a finite number."
        return None
    if not lo <= value <= hi:
        errors[key] = f"{label} must be between {lo} and {hi} (the model only covers New York City)."
        return None
    return value


def validate(values):
    """Return (clean_inputs, errors). errors maps field name -> message ('form' = general)."""
    errors = {}
    clean = {}

    clean["pickup_latitude"] = _parse_float(values.get("pickup_latitude"), "Pickup latitude", *LAT_RANGE, errors, "pickup_latitude")
    clean["pickup_longitude"] = _parse_float(values.get("pickup_longitude"), "Pickup longitude", *LON_RANGE, errors, "pickup_longitude")
    clean["dropoff_latitude"] = _parse_float(values.get("dropoff_latitude"), "Dropoff latitude", *LAT_RANGE, errors, "dropoff_latitude")
    clean["dropoff_longitude"] = _parse_float(values.get("dropoff_longitude"), "Dropoff longitude", *LON_RANGE, errors, "dropoff_longitude")

    # Passenger count: whole number within range
    raw_p = str(values.get("passenger_count") or "").strip()
    if raw_p == "":
        errors["passenger_count"] = "Passenger count is required."
    else:
        try:
            p = float(raw_p)
            if not p.is_integer():
                raise ValueError
            p = int(p)
            if not PASSENGER_RANGE[0] <= p <= PASSENGER_RANGE[1]:
                errors["passenger_count"] = f"Passenger count must be between {PASSENGER_RANGE[0]} and {PASSENGER_RANGE[1]}."
            else:
                clean["passenger_count"] = p
        except ValueError:
            errors["passenger_count"] = "Passenger count must be a whole number."

    # Pickup date/time
    raw_dt = (values.get("pickup_datetime") or "").strip()
    if raw_dt == "":
        errors["pickup_datetime"] = "Pickup date and time is required."
    else:
        try:
            dt = datetime.fromisoformat(raw_dt)
            if not YEAR_MIN <= dt.year <= YEAR_MAX:
                errors["pickup_datetime"] = (
                    f"The model was trained on trips from {YEAR_MIN} to {YEAR_MAX}; "
                    f"please pick a date in that range."
                )
            else:
                clean["pickup_datetime"] = dt
        except ValueError:
            errors["pickup_datetime"] = "Could not read that date/time. Use the date picker."

    # Categorical choices must be one of the values seen in training
    for key, column in (("weather", "Weather"), ("traffic", "Traffic Condition"), ("car_condition", "Car Condition")):
        choice = (values.get(key) or "").strip()
        if choice == "":
            errors[key] = f"{column} is required."
        elif choice not in CATEGORIES[column]:
            errors[key] = f"'{choice}' is not a valid {column} option."
        else:
            clean[key] = choice

    # Cross-field check: trip length
    if not errors:
        km = float(up.haversine_km(clean["pickup_latitude"], clean["pickup_longitude"],
                                   clean["dropoff_latitude"], clean["dropoff_longitude"]))
        if km < MIN_TRIP_KM:
            errors["form"] = "Pickup and dropoff are the same place -- please enter two different locations."
        elif km > MAX_TRIP_KM:
            errors["form"] = f"That trip is {km:.0f} km long, which is outside the range this model supports (max {MAX_TRIP_KM:.0f} km)."
    return clean, errors


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_fare(clean):
    """Run validated raw inputs through the SAME pipeline used in training."""
    raw = pd.DataFrame([{
        "pickup_datetime": clean["pickup_datetime"],
        "pickup_latitude": clean["pickup_latitude"],
        "pickup_longitude": clean["pickup_longitude"],
        "dropoff_latitude": clean["dropoff_latitude"],
        "dropoff_longitude": clean["dropoff_longitude"],
        "passenger_count": clean["passenger_count"],
        "Car Condition": clean["car_condition"],
        "Weather": clean["weather"],
        "Traffic Condition": clean["traffic"],
    }])
    features = up.add_features(raw, BEARING_CONVENTION)       # feature engineering
    fare = float(MODEL.predict(features)[0])                  # winsorize/encode/scale/model
    return max(fare, 0.0), features.iloc[0]


def trip_summary(clean, features):
    dt = clean["pickup_datetime"]
    return {
        "distance_km": round(float(features["distance"]), 2),
        "when": dt.strftime("%A, %d %b %Y at %H:%M"),
        "passengers": clean["passenger_count"],
        "conditions": f'{clean["weather"].title()} / {clean["traffic"]} / {clean["car_condition"]} car',
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
def render_page(values, errors=None, result=None):
    return render_template(
        "index.html", values=values, errors=errors or {}, result=result,
        categories=CATEGORIES, presets=LANDMARK_PRESETS,
        model_name=META["model_name"], year_min=YEAR_MIN, year_max=YEAR_MAX,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_page(DEFAULT_VALUES)

    values = request.form.to_dict()
    clean, errors = validate(values)
    if errors:
        return render_page(values, errors=errors), 400
    try:
        fare, features = predict_fare(clean)
    except Exception:  # never show a stack trace to the user
        app.logger.exception("Prediction failed")
        return render_page(values, errors={"form": "Something went wrong while predicting. Please check your inputs and try again."}), 500
    return render_page(values, result={"fare": f"{fare:,.2f}", **trip_summary(clean, features)})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"errors": {"form": "Send a JSON object as the request body."}}), 400
    values = {k: ("" if v is None else str(v)) for k, v in payload.items()}
    clean, errors = validate(values)
    if errors:
        return jsonify({"errors": errors}), 400
    fare, features = predict_fare(clean)
    return jsonify({"predicted_fare_usd": round(fare, 2),
                    "distance_km": round(float(features["distance"]), 2),
                    "model": META["model_name"]})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": META["model_name"]})


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)

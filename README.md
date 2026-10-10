# 🚕 Uber Fare Prediction

End-to-end machine learning project that predicts the fare of a New York City Uber ride — from exploratory data analysis, through a leak-free preprocessing pipeline, to a deployed web app.

**🔗 Live demo: [uber-fare.vercel.app](https://uber-fare.vercel.app/)**

---

## Overview

| Stage | Folder | What it does |
|---|---|---|
| 1. Exploratory Data Analysis | [`EDA/`](EDA) | Explores ~58K real NYC trips and picks the right chart for each question |
| 2. Data Preprocessing | [`Data_Preprocessing/`](Data_Preprocessing) | Cleans, encodes and scales the data in a reusable scikit-learn pipeline, then trains baseline and tuned regression models |
| 3. Deployment | [`Deployment/`](Deployment) | Compares four models, saves the best one, and serves it through a web app |

**Target variable:** `fare_amount` (USD)
**Dataset:** one row per trip — pickup/dropoff coordinates, pickup time, passenger count, weather, traffic and car condition, plus derived distance features.

---

## 1. Exploratory Data Analysis

Eight questions, each answered with the chart type that fits the data (scatter, boxplot, line, count plot).

| Question | Chart | Finding |
|---|---|---|
| Does passenger count affect fare? | Scatter | No — correlation ≈ 0.018 |
| Does car condition affect fare? | Boxplot | No — identical $8.50 median across all conditions |
| When are fares highest? | Line | Average fare peaks around 5 AM (about $15.30), a small-sample effect |
| Does traffic affect fare? | Boxplot | No — about $0.08 difference |
| Is distance the strongest predictor? | Scatter | Surprisingly weak (≈ 0.018), because of extreme outliers |
| When is demand highest? | Count plot | Peaks at 7 PM (3,532 requests) vs. 559 at 5 AM |
| Does weather change trip distance? | Boxplot | Medians nearly identical; mean differences are outlier-driven |
| Are airport-area pickups pricier? | Scatter | No — correlation ≈ 0.008 for JFK |

**Key insight:** the `distance` column reaches **8,670 km** against a **2.2 km median** — impossible for an NYC trip. Those outliers hide the real distance–fare relationship, so handling them became the most important cleaning step. Fares are also right-skewed (mean $11.35 vs. median $8.50).

---

## 2. Data Preprocessing

Every decision is justified by an EDA finding.

| Step | Decision | Why |
|---|---|---|
| Identifier columns | Dropped `User ID`, `User Name`, `Driver Name` | No predictive value |
| Missing values | `dropna()` | Under ~3% missing, consistent with MCAR |
| Duplicates | Dropped exact and `key`-based duplicates | `key` is a unique trip ID |
| Invalid values | Dropped fare ≤ 0 or distance ≤ 0 | Data errors, not real trips |
| Distance outliers | Winsorized at the 1st/99th percentile, **fit on train only** | Impossible values, but the rest of the row is valid |
| `hour` | Encoded cyclically (`hour_sin`, `hour_cos`) | Hour 23 and hour 0 are neighbours |
| `Car Condition` | Ordinal encoding (Bad < Good < Very Good < Excellent) | Real quality order |
| `Weather`, `Traffic Condition` | One-hot encoding | Low cardinality, no natural order |
| Numeric features | `RobustScaler` | Robust to remaining outliers |
| Target | `log1p` on train, `expm1` on predictions | Right-skewed fare |
| Split | 80/20 **before** any fitting | Prevents data leakage |

Everything that learns a statistic (caps, encoders, scaler) sits inside one scikit-learn `Pipeline`, so each cross-validation fold refits it on its own training portion only.

**Models:** Ridge Regression baseline with 5-fold CV, and a Random Forest tuned with `RandomizedSearchCV`.

---

## 3. Deployment

### Model comparison

Four regression models were trained on the same pipeline and evaluated on the held-out test set:

| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| Linear Regression | [ ] | [ ] | [ ] |
| Decision Tree | [ ] | [ ] | [ ] |
| Random Forest | [ ] | [ ] | [ ] |
| Gradient Boosting | [ ] | [ ] | [ ] |

**Selected model: [model name]** — chosen on test-set RMSE. When models were within 2% of each other, the simpler and faster one was preferred.

### The web app

- Form with route (pickup/dropoff coordinates, with NYC landmark quick-fill), trip time, passengers, weather, traffic and car condition
- Server-side validation with clear messages (required fields, coordinates outside NYC, same pickup and dropoff, dates outside the training years, unknown categories)
- Raw inputs pass through the **same feature engineering and pipeline used in training**, so predictions match what the model learned
- Model and preprocessing objects are loaded once at startup

### JSON API

```bash
curl -X POST https://uber-fare.vercel.app/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pickup_latitude": 40.758, "pickup_longitude": -73.9855,
    "dropoff_latitude": 40.6413, "dropoff_longitude": -73.7781,
    "passenger_count": 1, "pickup_datetime": "2014-06-15T18:30",
    "weather": "sunny", "traffic": "Flow Traffic", "car_condition": "Good"
  }'
```

Invalid input returns HTTP 400 with a message per field.

---

## Run locally

```bash
git clone https://github.com/dyajaballh8/Uber_Fare.git
cd Uber_Fare/Deployment

pip install -r requirements.txt

# Place final_internship_data.csv in this folder, then run every cell of the
# training notebook. It creates the saved model files the app needs.
jupyter notebook

python app.py          # open http://127.0.0.1:5000
```

Train and serve with the **same scikit-learn version**, since saved pickles are version-sensitive.

---

## Tech stack

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Flask · joblib · Vercel

---

## Notes and limitations

- The model was trained on NYC trips from a fixed range of years, so the app rejects dates and coordinates outside that range rather than extrapolating.
- Distance features use the haversine formula in kilometres, measured to JFK, EWR, LGA, the Statue of Liberty and the NYC centre.
- Predictions are estimates, not price quotes. Real Uber pricing includes surge and other factors this dataset does not capture.

---

## Author

**Dyaa Abdullah** · [LinkedIn](https://www.linkedin.com/in/your-profile) · [GitHub](https://github.com/dyajaballh8)

Built during the ML Internship Program at Cellula Technologies.

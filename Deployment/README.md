# Uber Fare Predictor — Task 3 (Model Deployment with Flask)

A Flask web app that predicts an NYC Uber fare from raw trip details. Four regression models
(Linear Regression, Decision Tree, Random Forest, Gradient Boosting) are compared in a notebook;
the model selected by test-set evidence is saved and served.

## Project files

| File | Purpose |
|---|---|
| `train_and_compare_models.ipynb` | Parts A + B: trains and compares the 4 models, picks one, saves it |
| `uber_pipeline.py` | The Task 2 pipeline (feature engineering + winsorize/encode/scale/log-target). Imported by **both** the notebook and the app |
| `app.py` | Part C: Flask app (HTML form, validation, `/api/predict`) |
| `templates/index.html` | The UI |
| `requirements.txt` | Python dependencies |
| `uber_fare_model.pkl` | *(created by the notebook)* full fitted Pipeline: preprocessing + model |
| `uber_fare_preprocessor.pkl` | *(created by the notebook)* the fitted preprocessor on its own (scaler/encoders) |
| `model_meta.json` | *(created by the notebook)* dropdown categories, training year range, bearing convention |

## How to run locally

```bash
# 1. install dependencies (Python 3.9+)
pip install -r requirements.txt

# 2. put final_internship_data.csv in this folder, then run ALL cells of the notebook
jupyter notebook train_and_compare_models.ipynb
#    -> creates uber_fare_model.pkl, uber_fare_preprocessor.pkl, model_meta.json

# 3. start the app
python app.py
#    -> open http://127.0.0.1:5000
```

Train and serve with the **same scikit-learn version** (pickles are version-sensitive). If `app.py` says
"Model files not found", step 2 has not been run yet.

## How the Task 2 pipeline is applied to user input

```
form values
  -> validate()                      required / numeric / range / category checks
  -> uber_pipeline.add_features()    distance, bearing, airport distances, cyclical hour, is_weekend, day/month/year
  -> saved Pipeline.predict()        winsorize -> encode (ordinal + one-hot) -> RobustScaler -> model -> expm1
  -> predicted fare ($)
```

The model and metadata are loaded **once** at startup, never retrained per request.

## Input handling

* HTML: required fields marked `*`, number inputs, a `datetime-local` picker, dropdowns for categorical fields, sensible defaults.
* Server (`validate()` in `app.py`): empty fields, non-numbers, NaN/inf, coordinates outside NYC, passenger count not a whole number 1–6,
  unparseable dates, dates outside the training years, unknown categories, pickup = dropoff, and trips over 100 km. Each problem produces
  a clear message next to the field and in a banner; the user's entries are kept. Unexpected failures show a friendly message, never a stack trace.

## Test cases for Part D (do these through the UI and screenshot each)

| # | Pickup | Dropoff | Passengers / time | Expect |
|---|---|---|---|---|
| 1 | Times Square | JFK Airport | 1, any weekday evening | prediction shown |
| 2 | Times Square | Central Park | 2, weekday morning | prediction shown (short trip, low fare) |
| 3 | JFK Airport | LaGuardia Airport | 3, weekend night | prediction shown |
| 4 (invalid) | Pickup latitude `10` | any | any | red banner + message under the field, no prediction |
| 5 (invalid, optional) | same location for pickup and dropoff | | | "Pickup and dropoff are the same place" |
| 6 (invalid, optional) | any trip, date in the year 2030 | | | "model was trained on trips from … to …" |

Use the quick-fill menus to drop landmarks into the coordinate fields. A completely empty required field is stopped by the
browser (`required`) before it reaches the server; the server checks it again, which you can see by calling the API (below).

## JSON API (optional)

```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d '{
  "pickup_latitude": 40.758, "pickup_longitude": -73.9855,
  "dropoff_latitude": 40.6413, "dropoff_longitude": -73.7781,
  "passenger_count": 1, "pickup_datetime": "2014-06-15T18:30",
  "weather": "sunny", "traffic": "Flow Traffic", "car_condition": "Good"}'
# -> {"distance_km": ..., "model": "...", "predicted_fare_usd": ...}

curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d '{"pickup_latitude": null}'
# -> 400 {"errors": {"pickup_latitude": "Pickup latitude is required.", ...}}
```
`GET /health` returns the loaded model name.

## Assumptions worth knowing

* Distances use the haversine formula in km, to these reference points: JFK (40.6413, -73.7781), EWR (40.6895, -74.1745),
  LGA (40.7769, -73.8740), Statue of Liberty (40.6892, -74.0445), NYC centre (40.7128, -74.0060).
  Notebook section 2 compares these recomputed values with the CSV's own columns so you can see how closely they agree.
* `day` is treated as day-of-month and `weekday` as day-of-week, as in Task 2.
* The model only knows New York trips from the years in the training data, so the app rejects coordinates and dates outside that range
  instead of extrapolating silently.

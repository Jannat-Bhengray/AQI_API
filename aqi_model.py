import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import os

df = pd.read_excel(r"C:\Users\vikas\Documents\indian_air_quality_dataset_10000.xlsx")
df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
df.drop_duplicates(inplace=True)
df.dropna(subset=["date", "aqi_value"], inplace=True)

for col in ["state", "area", "prominent_pollutants", "air_quality_status"]:
    df[col] = df[col].str.strip().str.title()

df = df[(df["aqi_value"] >= 0) & (df["aqi_value"] <= 500)]

df["number_of_monitoring_stations"] = (
    df.groupby(["state", "area"])["number_of_monitoring_stations"]
    .transform(lambda x: x.fillna(x.median()))
)
df["number_of_monitoring_stations"] = (
    df["number_of_monitoring_stations"]
    .fillna(df["number_of_monitoring_stations"].median()))
df["month"]       = df["date"].dt.month
df["year"]        = df["date"].dt.year
df["day_of_week"] = df["date"].dt.dayofweek
le_state     = LabelEncoder()
le_area      = LabelEncoder()
le_pollutant = LabelEncoder()

df["state_enc"]     = le_state.fit_transform(df["state"])
df["area_enc"]      = le_area.fit_transform(df["area"])
df["pollutant_enc"] = le_pollutant.fit_transform(df["prominent_pollutants"])
FEATURES = [
    "state_enc", "area_enc", "pollutant_enc",
    "number_of_monitoring_stations",
    "month", "year", "day_of_week"
]

X      = df[FEATURES]
y_reg  = df["aqi_value"]
y_clf  = df["air_quality_status"]

X_train, X_test, yr_train, yr_test, yc_train, yc_test = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape, "| Test size:", X_test.shape)
print("Any nulls in train?", X_train.isna().any().any())
print("\nTraining regressor...")
reg_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
reg_model.fit(X_train, yr_train)
yr_pred = reg_model.predict(X_test)
print("  MAE:", round(mean_absolute_error(yr_test, yr_pred), 2))
print("  R2 :", round(r2_score(yr_test, yr_pred), 3))
print("\nTraining classifier...")
clf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

clf_model.fit(X_train, yc_train)
yc_pred = clf_model.predict(X_test)
print("  Accuracy:", round(accuracy_score(yc_test, yc_pred), 3))
os.makedirs("models", exist_ok=True)
with open("models/reg_model.pkl", "wb") as f:
    pickle.dump(reg_model, f)

with open("models/clf_model.pkl", "wb") as f:
    pickle.dump(clf_model, f)

with open("models/le_state.pkl", "wb") as f:
    pickle.dump(le_state, f)

with open("models/le_area.pkl", "wb") as f:
    pickle.dump(le_area, f)

with open("models/le_pollutant.pkl", "wb") as f:
    pickle.dump(le_pollutant, f)

print("\nAll 5 pkl files saved in /models")
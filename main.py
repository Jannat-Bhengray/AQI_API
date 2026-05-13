import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Loading all pkl files  
with open("models/reg_model.pkl", "rb") as f:
    reg_model = pickle.load(f)

with open("models/clf_model.pkl", "rb") as f:
    clf_model = pickle.load(f)

with open("models/le_state.pkl", "rb") as f:
    le_state = pickle.load(f)

with open("models/le_area.pkl", "rb") as f:
    le_area = pickle.load(f)

with open("models/le_pollutant.pkl", "rb") as f:
    le_pollutant = pickle.load(f)

#  App    
app = FastAPI(title="AQI Prediction API")

#   Input 
class AQIInput(BaseModel):
    state: str                         # e.g. "Delhi"
    area: str                          # e.g. "Anand Vihar"
    prominent_pollutant: str           # e.g. "Pm2.5"
    number_of_monitoring_stations: int # e.g. 5
    month: int                         # 1 to 12
    year: int                          # e.g. 2023
    day_of_week: int                   # 0=Mon to 6=Sun

#  Safe encoder helper  
def encode(encoder, value: str, field_name: str):
    value = value.strip().title()
    if value not in encoder.classes_:
        raise HTTPException(
            status_code=422,
            detail=f"'{value}' is not a valid {field_name}. "
                   f"Visit /options to see all valid values."
        )
    return int(encoder.transform([value])[0])

#   Endpoints  

@app.get("/")
def home():
    return {"message": "AQI API is running. Go to /docs to test it."}


@app.get("/options")
def options():
    # shows every valid value the user can pass
    return {
        "states":     sorted(le_state.classes_.tolist()),
        "areas":      sorted(le_area.classes_.tolist()),
        "pollutants": sorted(le_pollutant.classes_.tolist())
    }


@app.post("/predict")
def predict(data: AQIInput):

    # encode categorical fields
    state_enc     = encode(le_state,     data.state,               "state")
    area_enc      = encode(le_area,      data.area,                "area")
    pollutant_enc = encode(le_pollutant, data.prominent_pollutant, "pollutant")

    # build feature list — same order as training
    features = [[
        state_enc,
        area_enc,
        pollutant_enc,
        data.number_of_monitoring_stations,
        data.month,
        data.year,
        data.day_of_week
    ]]

    # predict
    aqi_value  = round(float(reg_model.predict(features)[0]), 2)
    aqi_status = clf_model.predict(features)[0]

    return {
        "aqi_value":          aqi_value,
        "air_quality_status": aqi_status
    }
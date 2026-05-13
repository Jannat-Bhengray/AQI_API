#  AQI Prediction API

An end-to-end Machine Learning + FastAPI project that predicts:

- AQI Value (Regression)
- Air Quality Status (Classification)

using environmental and regional air quality data.

 

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Pydantic
- Pickle

 

# Features

- Data cleaning & preprocessing
- Feature engineering from date columns
- Random Forest Regressor for AQI prediction
- Random Forest Classifier for AQI status prediction
- Label encoding for categorical features
- Input validation & error handling
- Interactive Swagger API docs
- Serialized ML models using `.pkl`

 

# Dataset Processing

Performed:
- duplicate removal
- missing value handling
- AQI range filtering (0–500)
- categorical standardization
- temporal feature extraction:
  - month
  - year
  - day_of_week

Tested seasonal feature engineering during EDA, but final model performed better with month/day-based features.

 

# API Endpoints

## `GET /`
Health check endpoint.

## `GET /options`
Returns all valid:
- states
- areas
- pollutants

## `POST /predict`
Predicts:
- AQI value
- Air quality status

### Sample Input

```json
{
  "state": "Delhi",
  "area": "Anand Vihar",
  "prominent_pollutant": "Pm2.5",
  "number_of_monitoring_stations": 5,
  "month": 7,
  "year": 2023,
  "day_of_week": 2
}
```

### Sample Output

```json
{
  "aqi_value": 287.41,
  "air_quality_status": "Poor"
}
 

 

# Run Locally
 
Open Swagger Docs:

```bash
http://127.0.0.1:8000/docs
```

 

# Future Improvements

- Model comparison (XGBoost/CatBoost)
- Cloud deployment by docker
- Feature importance visualization
- Frontend dashboard

 

# Author

Jannat Bhengray 

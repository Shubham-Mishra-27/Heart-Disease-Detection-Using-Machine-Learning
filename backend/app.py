from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI()

# Load model
model = pickle.load(open("lgb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

class PatientData(BaseModel):
    age: float
    gender: int
    height: float
    weight: float
    ap_hi: float
    ap_lo: float
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int

@app.get("/")
def home():
    return {"message": "API working"}

@app.post("/predict")
def predict(data: PatientData):
    input_data = np.array([[
        data.age, data.gender, data.height, data.weight,
        data.ap_hi, data.ap_lo, data.cholesterol,
        data.gluc, data.smoke, data.alco, data.active
    ]])

    input_scaled = input_data  # ❗ bypass scaler

    prob = model.predict_proba(input_scaled)[0][1]
    prediction = int(prob > 0.4)

    return {
        "prediction": prediction,
        "risk": float(prob)
    }




app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
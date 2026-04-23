# ❤️ CardioHealth - Heart Disease Prediction System

This project is a web-based application that predicts the risk of heart disease using machine learning. It allows users to enter basic health details and instantly receive a risk score along with a clear classification (High Risk / Low Risk).

The system combines a user-friendly frontend with a powerful backend model to make health assessment simple and accessible.

---

## 📌 Features

- Predicts heart disease risk using trained ML model (LightGBM)
- Clean and simple user interface
- Real-time prediction results
- Uses basic health inputs (age, BP, cholesterol, etc.)
- Backend powered by FastAPI
- Frontend built with React + Tailwind CSS
- Scalable and easy to use

---

## ⚙️ Tech Stack

- **Frontend:** React, Tailwind CSS  
- **Backend:** FastAPI (Python)  
- **Machine Learning:** LightGBM, Scikit-learn  
- **Other Tools:** NumPy, Pandas  

---

## 🧠 How It Works

1. User enters health details  
2. Data is processed and converted  
3. Model predicts probability of heart disease  
4. Result is displayed instantly  

---

## 📊 Model Details

- Algorithm: LightGBM  
- Input Features:
  - Age, Gender  
  - Height, Weight  
  - Blood Pressure (ap_hi, ap_lo)  
  - Cholesterol, Glucose  
  - Smoking, Alcohol  
  - Physical Activity  

---

## 🚀 Setup

### 1. Clone repository
git clone https://github.com/Shubham-Mishra-27/Heart-Disease-Detection-Using-Machine-Learning.git

cd Heart-Disease-Detection-Using-Machine-Learning


### 2. Run Backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload

Backend will run on:
http://127.0.0.1:8000

### 3. Run Frontend
cd frontend
npm install
npm start

Frontend will run on:
http://localhost:3000
```bash

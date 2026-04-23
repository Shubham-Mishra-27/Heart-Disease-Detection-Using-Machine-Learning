
# !pip install scikit-learn pandas numpy
# !pip install xgboost
# !pip install shap
# !pip install lightgbm

#library
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



df = pd.read_csv("model/cardio_train.csv", sep=";")


if "id" in df.columns:
    df = df.drop("id", axis=1)


if df["age"].max() > 200:
    df["age"] = (df["age"] / 365).astype(int)
print(df.head())
# target column handling
target_col = "cardio" if "cardio" in df.columns else "target"


df = df.replace("?", np.nan)
df = df.dropna()
df = df.apply(pd.to_numeric)


print("Dataset shape:", df.shape)

X = df.drop(target_col, axis=1)
y = df[target_col]


df = df[(df["ap_hi"] >= 80) & (df["ap_hi"] <= 200)]
df = df[(df["ap_lo"] >= 40) & (df["ap_lo"] <= 140)]



df = df[(df["height"] >= 120) & (df["height"] <= 220)]
df = df[(df["weight"] >= 35) & (df["weight"] <= 200)]

df["BMI"] = df["weight"] / ((df["height"]/100) ** 2)
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = RandomForestClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)



y_pred = model.predict(X_test_scaled)

print("\n=====RANDOM FOREST MODEL PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

model = XGBClassifier(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=5,
    gamma=0.2,
    reg_lambda=2,
    reg_alpha=0.2,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("\n===== XGBOOST PERFORMANCE =====")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

lgb_model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=5,
    random_state=42
)

lgb_model.fit(X_train, y_train)




from sklearn.metrics import accuracy_score, classification_report

lgb_pred = lgb_model.predict(X_test)

print("LightGBM Accuracy:", accuracy_score(y_test, lgb_pred))
print(classification_report(y_test, lgb_pred))

with open("backend/heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("backend/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\n✅ Model saved as heart_model.pkl")
print("✅ Scaler saved as scaler.pkl")

import pickle
import numpy as np
import pandas as pd

# load model and scaler

pickle.dump(lgb_model, open("backend/lgb_model.pkl", "wb"))
model = pickle.load(open("backend/heart_model.pkl", "rb"))
scaler = pickle.load(open("backend/scaler.pkl", "rb"))

feature_names = [
    "age","gender","height","weight","ap_hi","ap_lo",
    "cholesterol","gluc","smoke","alco","active"
]

user_data = {
    "age": 65,
    "gender": 1,
    "height": 165,
    "weight": 95,
    "ap_hi": 170,
    "ap_lo": 105,
    "cholesterol": 3,
    "gluc": 3,
    "smoke": 1,
    "alco": 1,
    "active": 0
}

# convert to dataframe
input_df = pd.DataFrame([user_data])

# derived features
input_df["BMI"] = input_df["weight"] / ((input_df["height"]/100) ** 2)
input_df["pulse_pressure"] = input_df["ap_hi"] - input_df["ap_lo"]
input_df["MAP"] = (input_df["ap_hi"] + 2*input_df["ap_lo"]) / 3

# reorder columns
input_df = input_df[feature_names]

input_scaled = scaler.transform(input_df)

risk_prob = model.predict_proba(input_scaled)[0][1]
prediction = 1 if risk_prob > 0.46 else 0

print("\n===== PREDICTION =====")

if prediction == 1:
    print("⚠️ High Risk of Heart Disease")
else:
    print("✅ Low Risk of Heart Disease")

print(f"Risk Probability: {risk_prob:.2f}")

# @title
# feature importance from model
importances = model.feature_importances_

# contribution approximation
contrib = input_df.iloc[0] * importances

top_reason = contrib.abs().sort_values(ascending=False).index[0]

print("\n Main Risk Driver:", top_reason)

import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_scaled)

# get feature contributions
vals = shap_values[0]

# find top feature
top_feature = input_df.columns[abs(vals).argmax()]

print("Main Reason:", top_feature)





import matplotlib.pyplot as plt

models = ['XGBoost', 'LightGBM']
scores = [
    accuracy_score(y_test, model.predict(X_test)),
    accuracy_score(y_test, lgb_model.predict(X_test))
]

plt.figure()
plt.bar(models, scores)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()

import pandas as pd

importance = lgb_model.feature_importances_
features = X.columns

df = pd.DataFrame({'Feature': features, 'Importance': importance})
df = df.sort_values(by='Importance', ascending=False)

plt.figure()
plt.barh(df['Feature'], df['Importance'])
plt.title("LightGBM Feature Importance")
plt.xlabel("Importance")
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, lgb_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix - LightGBM")
plt.show()

from sklearn.metrics import roc_curve, auc

y_prob = lgb_model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - LightGBM")
plt.legend()
plt.show()

plt.figure()
plt.hist(y_prob, bins=30)
plt.title("Prediction Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.show()

import shap

explainer = shap.TreeExplainer(lgb_model)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test)



# from pyngrok import ngrok
# import uvicorn
# import threading

# def run():
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# threading.Thread(target=run).start()

# public_url = ngrok.connect(8000)
# print(public_url)


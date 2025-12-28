import streamlit as st
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Cardiac Compass", layout="centered")
st.title("Cardiac Compass – Heart Disease Prediction")

@st.cache_resource
def train_model():
    df = pd.read_csv("heart_data.csv")

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df.columns = df.columns.str.strip()
    df.fillna(df.median(numeric_only=True), inplace=True)

    if "sex" in df.columns:
        df["sex"] = df["sex"].map({1: "male", 0: "female"})

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=50,
        random_state=42
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return model, X.columns, acc


model, feature_names, accuracy = train_model()
st.success(f"Model trained | Accuracy: {accuracy*100:.2f}%")

st.subheader("Enter Patient Details")

user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(feature, value=0.0)

input_df = pd.DataFrame([user_input])

if st.button("Predict"):
    result = model.predict(input_df)[0]
    if result == 1:
        st.error("High risk of heart disease")
    else:
        st.success("Low risk of heart disease")

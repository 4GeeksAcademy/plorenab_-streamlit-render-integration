import streamlit as st
from pickle import load


# Cargar el modelo
model = load(open("../models/knn_classifier_k10_manhattan_distance.sav", "rb"))

class_dict = {
    0: "Calidad baja",
    1: "Calidad Media",
    2: "Calidad Alta"
}

st.title("Wine Quality - Model Prediction")

st.write("Introduce the values for wine characteristics")

fixed_acidity = st.number_input("Fixed acidity", min_value=0.0, max_value=20.0, step=0.1)
volatile_acidity = st.number_input("Volatile acidity", min_value=0.0, max_value=2.0, step=0.01)
citric_acid = st.number_input("Citric acid", min_value=0.0, max_value=1.0, step=0.01)
residual_sugar = st.number_input("Residual sugar", min_value=0.0, max_value=20.0, step=0.1)
chlorides = st.number_input("Chlorides", min_value=0.0, max_value=0.2, step=0.001)
free_sulfur = st.number_input("Free sulfur dioxide", min_value=0.0, max_value=100.0, step=1.0)
total_sulfur = st.number_input("Total sulfur dioxide", min_value=0.0, max_value=300.0, step=1.0)
density = st.number_input("Density", min_value=0.985, max_value=1.020, step=0.0001)
ph = st.number_input("pH", min_value=2.5, max_value=4.5, step=0.01)
sulphates = st.number_input("Sulphates", min_value=0.0, max_value=2.0, step=0.01)
alcohol = st.number_input("Alcohol", min_value=5.0, max_value=15.0, step=0.1)

if st.button("Predict"):
    X = [[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur, total_sulfur, density, ph, sulphates, alcohol
    ]]
    prediction = int(model.predict(X)[0])
    pred_class = class_dict.get(prediction, "Calidad desconocida")

    st.subheader("Resultado de la predicción")
    st.write(f'**{pred_class}** (clase {prediction})')
import gradio as gr
import numpy as np
import pandas as pd
import joblib
import requests

model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")
safe_features = ["humidity", "pressure", "temp_mean", "temp_max", "temp_min"]

def fetch_weather_data(lat, lon):
    API_KEY = "17b2877bc8114065a27174823251805"
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={lat},{lon}"
    res = requests.get(url).json()

    temp_max = res['current']['temp_c']
    temp_min = temp_max - 2
    temp_mean = (temp_max + temp_min) / 2

    weather_data = {
        "temp_max": temp_max,
        "temp_min": temp_min,
        "temp_mean": temp_mean,
        "humidity": res['current']['humidity'],
        "pressure": res['current']['pressure_mb'] / 10,
    }
    return weather_data

def xyz(mode, temp_min, temp_max, temp_mean, humidity, pressure, lat, lon):
    if mode == "Auto":
        if lat is None or lon is None:
            return "Please provide lat and lon"
        data = fetch_weather_data(lat, lon)
    else:
        data = {
            "temp_max": temp_max,
            "temp_min": temp_min,
            "temp_mean": temp_mean,
            "humidity": humidity,
            "pressure": pressure,
        }

    for feature in safe_features:
        if feature not in data or data[feature] is None:
            return "Missing data in input."

    X = np.array([[data[f] for f in safe_features]])
    y_pred = model.predict(X)
    return le.inverse_transform(y_pred)[0]

interface = gr.Interface(
    fn=xyz,
    inputs=[
        gr.Radio(["Manual", "Auto"], label="Mode"),
        gr.Number(label="Temp Min (°C)", value=20),
        gr.Number(label="Temp Max (°C)", value=25),
        gr.Number(label="Temp Mean (°C)", value=22.5),
        gr.Number(label="Humidity (%)", value=60),
        gr.Number(label="Pressure (kPa)", value=101.3),
        gr.Number(label="Latitude", value=None),
        gr.Number(label="Longitude", value=None),
    ],
    outputs="text",
    title="Weather Condition Predictor",
    description="Predicts weather condition using either manual or automatic weather data input."
)
interface.launch(share=True)

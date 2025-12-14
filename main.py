from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model_knn.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction_text=None)



@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = scaler.transform([features])
    prediction = model.predict(final_features)

    if prediction[0] == 1:
        hasil = "Pelanggan DIPREDIKSI akan berlangganan deposito"
    else:
        hasil = "Pelanggan DIPREDIKSI tidak berlangganan deposito"

    return render_template("index.html", prediction_text=hasil)


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

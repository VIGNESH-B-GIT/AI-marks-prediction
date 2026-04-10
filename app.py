from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

history = []

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    hours = float(request.form["hours"])
    prediction = model.predict([[hours]])[0]
    result = round(prediction, 2)

    history.append((hours, result))

    return render_template("index.html", result=result, history=history)

if __name__ == "__main__":
    app.run(debug=True)
import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port, debug=False)
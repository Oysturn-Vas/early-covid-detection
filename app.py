from flask import Flask, request, render_template
import numpy as np
from joblib import load

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test")
def test():
    return render_template("test.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/result", methods=["POST"])
def result():
    cough = int(request.form.get("cough"))
    fever = int(request.form.get("fever"))
    sore_throat = int(request.form.get("sore_throat"))
    sob = int(request.form.get("sob"))
    head_ache = int(request.form.get("head_ache"))
    age = int(request.form.get("age"))
    contact = int(request.form.get("contact"))
    gender = int(request.form.get("gender"))
    abroad = int(request.form.get("abroad"))

    prediction = np.array(
        [[cough, fever, sore_throat, sob, head_ache, age, contact, gender, abroad]])
    model = load('model.joblib')
    preds = model.predict(prediction)
    return render_template("result.html", data = preds[0])


if __name__ == "__main__":
    app.run(debug=True)

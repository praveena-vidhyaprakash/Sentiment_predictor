from flask import Flask, request, render_template
import joblib
import os
import pandas as pd

app = Flask(__name__)

MODEL_PATH = "sentiment_pipeline.pkl"
if not os.path.exists(MODEL_PATH):
    import train  # will create model if missing

pipeline = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, review_text="", star_rating=3)

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("review_text", "").strip()
    star = request.form.get("star_rating", "")
    try:
        star_val = float(star) if star.strip() != "" else 3.0
    except:
        star_val = 3.0
    review_len = len(text.split())
    df = pd.DataFrame([{
        "review_text": text,
        "star_rating": star_val,
        "review_len": review_len
    }])
    proba = pipeline.predict_proba(df)[0, 1]
    sentiment = int(pipeline.predict(df)[0])
    result = {
        "sentiment": sentiment,
        "proba": proba,
        "review_len": review_len
    }
    return render_template("index.html", result=result, review_text=text, star_rating=star_val)

if __name__ == "__main__":
    app.run(debug=True)

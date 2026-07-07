from flask import Flask, render_template, request , send_file
import pickle
import pandas as pd

# Create Flask App
app = Flask(__name__)

# Load Saved Model and TF-IDF Vectorizer
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("vectorizer.pkl", "rb"))

# Store Previous Predictions
history = []

# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html", history=history)

# -----------------------------
# Single Review Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # Get review from HTML form
    review = request.form["review"]

    # Convert review into TF-IDF vector
    review_vec = tfidf.transform([review])

    # Predict Sentiment
    prediction = model.predict(review_vec)[0]

    # Prediction Confidence
    confidence = max(model.predict_proba(review_vec)[0]) * 100

    # Save Prediction in History
    history.insert(0, {
        "review": review,
        "prediction": prediction,
        "confidence": round(confidence, 2)
    })

    # Send result back to HTML
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence, 2),
        review=review,
        history=history
    )

# -----------------------------
# Clear History
# -----------------------------
@app.route("/clear")
def clear_history():

    history.clear()

    return render_template(
        "index.html",
        history=history
    )
@app.route("/download")
def download():

    return send_file(
        "analyzed_reviews.csv",
        as_attachment=True
    )
# -----------------------------
# CSV Upload
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload_csv():

    # Get uploaded CSV
    csv_file = request.files["csv_file"]

    # Read CSV
    df = pd.read_csv(csv_file)

    # Get review column
    reviews = df["review_text"]

    # Convert reviews into TF-IDF vectors
    review_vectors = tfidf.transform(reviews)

    # Predict sentiments
    predictions = model.predict(review_vectors)

    # Prediction confidence
    confidences = model.predict_proba(review_vectors).max(axis=1) * 100

    # Add new columns
    df["Predicted Sentiment"] = predictions
    df["Confidence"] = confidences.round(2)

    total_reviews = len(df)
    positive_count = (df["Predicted Sentiment"] == "Positive").sum()
    negative_count = (df["Predicted Sentiment"] == "Negative").sum()
    neutral_count = (df["Predicted Sentiment"] == "Neutral").sum()
    average_confidence = round(df["Confidence"].mean(), 2)

    # Save analyzed CSV
    output_file = "analyzed_reviews.csv"

    df.to_csv(output_file, index=False)

    return render_template(
        "index.html",
        total_reviews=total_reviews,
        positive_count=positive_count,
        negative_count=negative_count,
        neutral_count=neutral_count,
        average_confidence=average_confidence,
        download_ready=True,
        history=history
    )

# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
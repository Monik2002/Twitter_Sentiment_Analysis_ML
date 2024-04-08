from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

app = Flask(__name__)
CORS(app)

# Load the dataset and train the model
df = pd.read_csv('Data.csv')
tfidf = TfidfVectorizer(stop_words='english')
X = tfidf.fit_transform(df['review'])
y = df['sentiment']
svc = SVC(C=1, kernel='linear')
svc.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    if request.method == 'POST':
        # Getting the review text from the request
        review_text = request.json.get('review')

        # Vectorize the review text using the TF-IDF vectorizer
        review_vector = tfidf.transform([review_text])

        # Predicting the sentiment of the review
        sentiment = svc.predict(review_vector)[0]

        # Returning the predicted sentiment
        return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)


# run with python app.py
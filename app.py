
from flask import Flask, render_template, request
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load and preprocess data
df = pd.read_excel('Movies Data.xlsx')  # Make sure this file is in the same folder

# Drop unnecessary columns and clean data
df.drop('vote_count', inplace=True, axis=1)
df['runtime'] = df['runtime'].fillna(0)
df = df[df['vote_average'] != 0]
df = df[df['runtime'] != 0]

# Exclude non-English or foreign language films
df = df[~df['original_language'].isin(['ja', 'ko', 'cn', 'zh', 'ru', 'hi', 'th', 'pl', 'uk', 'fa', 'is', 'ar'])]

# Define feature columns (ensure these exist in your dataset)
features = ["popularity", "vote_average", "runtime", "Drama & Emotion", "Entertainment", "Thriller & Mystery"]
X = df[features]

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train Nearest Neighbors model
knn = NearestNeighbors(n_neighbors=1, metric='euclidean')
knn.fit(X_scaled)

# Define recommendation function
def recommend_movie(min_rating, category, runtime):
    category_vector = [1 if category == cat else 0 for cat in ["Drama & Emotion", "Entertainment", "Thriller & Mystery"]]
    query_vector = pd.DataFrame([[100, min_rating, runtime] + category_vector], columns=features)
    query_scaled = scaler.transform(query_vector)
    distance, index = knn.kneighbors(query_scaled, n_neighbors=1)
    recommended_movie = df.iloc[index[0][0]][["id", "original_title", "tagline", "overview", "runtime", "vote_average", "popularity"]]
    recommended_movie["runtime"] = f"{recommended_movie['runtime']} minutes"
    return recommended_movie

# Define the main route for form submission
@app.route('/', methods=['GET', 'POST'])
def index():
    movie = None
    if request.method == 'POST':
        min_rating = int(request.form['rating'])
        cate = int(request.form['category'])
        runtime = int(request.form['runtime'])

        # Map numeric category to string
        if cate == 1:
            category = 'Drama & Emotion'
        elif cate == 2:
            category = 'Entertainment'
        else:
            category = 'Thriller & Mystery'

        movie = recommend_movie(min_rating, category, runtime).to_dict()

    return render_template('index.html', movie=movie)

if __name__ == '__main__':
    app.run(debug=True)


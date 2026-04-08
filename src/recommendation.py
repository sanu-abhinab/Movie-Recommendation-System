import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.utils import get_mood_keywords


class MovieRecommender:
    
    def recommend_by_mood(self, mood, top_n=5):
        mood_keywords = get_mood_keywords()

        if mood not in mood_keywords:
            return ["Invalid mood selected"]

        keywords = mood_keywords[mood]

        # Filter movies containing mood keywords
        filtered_movies = self.data[self.data['tags'].apply(
            lambda x: any(word in x.lower() for word in keywords)
        )]

        if filtered_movies.empty:
            return ["No movies found for this mood"]

        return filtered_movies['title'].head(top_n).tolist()

    def __init__(self, data_path=None):
        self.data = None
        self.similarity_matrix = None

        # If models already exist → load them
        if os.path.exists("models/similarity.pkl") and os.path.exists("models/movies.pkl"):
            self._load_model()
        else:
            self._build_model(data_path)

    def _build_model(self, data_path):
        print("Building model...")

        self.data = pd.read_csv(data_path)
        self.data['tags'] = self.data['tags'].fillna('')

        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.data['tags'])

        self.similarity_matrix = cosine_similarity(tfidf_matrix)

        # Save model
        self._save_model()

    def _save_model(self):
        os.makedirs("models", exist_ok=True)

        with open("models/similarity.pkl", "wb") as f:
            pickle.dump(self.similarity_matrix, f)

        with open("models/movies.pkl", "wb") as f:
            pickle.dump(self.data, f)

        print("Model saved successfully!")

    def _load_model(self):
        print("Loading model...")

        with open("models/similarity.pkl", "rb") as f:
            self.similarity_matrix = pickle.load(f)

        with open("models/movies.pkl", "rb") as f:
            self.data = pickle.load(f)

        print("Model loaded successfully!")

    def recommend(self, movie_title, top_n=5):
        if movie_title not in self.data['title'].values:
            return ["Movie not found"]

        index = self.data[self.data['title'] == movie_title].index[0]
        similarity_scores = list(enumerate(self.similarity_matrix[index]))

        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        recommendations = []
        for i in similarity_scores[1:top_n+1]:
            recommendations.append(self.data.iloc[i[0]].title)

        return recommendations
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.similarity_matrix = None
        self._prepare_data()

    def _prepare_data(self):
        # Fill missing tags
        self.data['tags'] = self.data['tags'].fillna('')

        # 🔥 Use TF-IDF instead of CountVectorizer
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english'
        )

        tfidf_matrix = tfidf.fit_transform(self.data['tags'])

        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(tfidf_matrix)

    def recommend(self, movie_title, top_n=5):
        if movie_title not in self.data['title'].values:
            return ["Movie not found in database"]

        movie_index = self.data[self.data['title'] == movie_title].index[0]
        similarity_scores = list(enumerate(self.similarity_matrix[movie_index]))

        # Sort by similarity score
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        recommended_movies = []
        for i in similarity_scores[1:top_n + 1]:
            recommended_movies.append(self.data.iloc[i[0]].title)

        return recommended_movies
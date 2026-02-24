import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:

    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.similarity_matrix = None
        self._prepare_data()

    def _prepare_data(self):
        # Fill missing values
        self.data['overview'] = self.data['overview'].fillna('')

        # Convert text into vectors
        vectorizer = CountVectorizer(stop_words='english')
        count_matrix = vectorizer.fit_transform(self.data['overview'])

        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(count_matrix)

    def recommend(self, movie_title, top_n=5):
        if movie_title not in self.data['title'].values:
            return ["Movie not found in database"]

        movie_index = self.data[self.data['title'] == movie_title].index[0]
        similarity_scores = list(enumerate(self.similarity_matrix[movie_index]))

        # Sort movies by similarity
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        recommended_movies = []
        for i in similarity_scores[1:top_n+1]:
            recommended_movies.append(self.data.iloc[i[0]].title)

        return recommended_movies
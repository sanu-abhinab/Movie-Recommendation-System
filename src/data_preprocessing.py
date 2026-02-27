import pandas as pd
import ast
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

class DataPreprocessor:
    
    def _stem(self, text):
        return " ".join([ps.stem(word) for word in text.split()])

    def __init__(self, movie_path, credits_path):
        self.movies = pd.read_csv(movie_path)
        self.credits = pd.read_csv(credits_path)

    def _convert(self, obj):
        """Extract name values from JSON-like string"""
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    def _get_director(self, obj):
        """Extract director name"""
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return i['name']
        return None

    def clean_data(self, save_path=None):
        # Merge datasets
        self.movies = self.movies.merge(self.credits, on='title')

        # Keep important columns only
        self.movies = self.movies[['title','overview','genres','keywords','cast','crew']]

        # Remove missing values
        self.movies.dropna(inplace=True)

        # Convert JSON columns
        self.movies['genres'] = self.movies['genres'].apply(self._convert)
        self.movies['keywords'] = self.movies['keywords'].apply(self._convert)
        self.movies['cast'] = self.movies['cast'].apply(self._convert)

        # Take only top 3 actors
        self.movies['cast'] = self.movies['cast'].apply(lambda x: x[:3])

        # Extract director
        self.movies['crew'] = self.movies['crew'].apply(self._get_director)

        # Convert overview into list
        self.movies['overview'] = self.movies['overview'].apply(lambda x: x.split())

        # Remove spaces in names
        self.movies['genres'] = self.movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
        self.movies['keywords'] = self.movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
        self.movies['cast'] = self.movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
        self.movies['crew'] = self.movies['crew'].apply(lambda x: x.replace(" ", "") if isinstance(x, str) else "")

        # Create tags column
        self.movies['tags'] = self.movies['overview'] + self.movies['genres'] + self.movies['keywords'] + self.movies['cast']
        self.movies['tags'] = self.movies['tags'].apply(lambda x: " ".join(x))
        self.movies['tags'] = self.movies['tags'].apply(self._stem)

        final_df = self.movies[['title','tags']]

        # ✅ Save cleaned data
        if save_path:
            final_df.to_csv(save_path, index=False)
            print(f"Cleaned data saved to {save_path}")

        return final_df
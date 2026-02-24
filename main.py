from src.recommendation import MovieRecommender

if __name__ == "__main__":
    recommender = MovieRecommender("data/processed/clean_movies.csv")

    movie_name = input("Enter movie name: ")
    recommendations = recommender.recommend(movie_name)

    print("\nRecommended Movies:")
    for movie in recommendations:
        print(movie)




#data cleaning
"""


from src.data_preprocessing import DataPreprocessor

if __name__ == "__main__":
    processor = DataPreprocessor(
        "data/raw/tmdb_5000_movies.csv",
        "data/raw/tmdb_5000_credits.csv"
    )

    clean_df = processor.clean_data(
        save_path="data/processed/clean_movies.csv"
    )

    print(clean_df.head())
"""
from src.recommendation import MovieRecommender
from src.tmdb_api import get_popular_movies, get_now_playing_movies

if __name__ == "__main__":
    recommender = MovieRecommender("data/processed/clean_movies.csv")

    print("\nChoose option:")
    print("1 → Similar Movies")
    print("2 → Mood-Based")
    print("3 → Mood + Movie")
    print("4 → Popular Movies (Live)")
    print("5 → Now Playing Movies")

    choice = input("Enter choice: ")

    if choice == "1":
        movie = input("Enter movie name: ")
        recs = recommender.recommend(movie)

    elif choice == "2":
        mood = input("Enter mood: ")
        recs = recommender.recommend_by_mood(mood)

    elif choice == "3":
        mood = input("Enter mood: ")
        movie = input("Enter movie name: ")
        recs = recommender.recommend_by_mood(mood, base_movie=movie)

    elif choice == "4":
        recs = get_popular_movies()

    elif choice == "5":
        recs = get_now_playing_movies()

    else:
        recs = ["Invalid choice"]

    print("\n🎬 Recommendations:")
    for r in recs:
        print(r)


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
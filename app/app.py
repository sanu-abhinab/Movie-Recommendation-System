import streamlit as st
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommendation import MovieRecommender

# Load recommender
recommender = MovieRecommender("data/processed/clean_movies.csv")

# App Title
st.title("🎬 Movie Recommendation System")

st.write("Get movie recommendations based on movies and mood!")

# Movie Selection
movie_list = recommender.data['title'].values

selected_movie = st.selectbox(
    "Select a movie",
    movie_list
)

# Mood Selection
mood_options = [
    "happy",
    "sad",
    "excited",
    "romantic",
    "scary",
    "motivational"
]

selected_mood = st.selectbox(
    "Select your mood",
    mood_options
)

# Recommendation Button
if st.button("Recommend"):

    recommendations = recommender.recommend_by_mood(
        mood=selected_mood,
        base_movie=selected_movie
    )

    st.subheader("🎥 Recommended Movies")

    for movie in recommendations:

        st.subheader(movie)

        explanation = recommender.explain_recommendation(
            selected_movie,
            movie
        )

        st.caption(explanation)
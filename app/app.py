import sys
import os


sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

import streamlit as st

from src.recommendation import MovieRecommender

from src.tmdb_api import (
    search_movie,
    get_movie_details,
    get_movie_credits,
    get_watch_providers,
    get_poster_url,
    get_watch_link
)

# Load recommender
recommender = MovieRecommender(
    "data/processed/clean_movies.csv"
)

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    font-size: 50px;
    font-weight: bold;
    color: #E50914;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: #AAAAAA;
    font-size: 20px;
    margin-bottom: 40px;
}

.movie-card {
    background-color: #1F1F1F;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    transition: 0.3s;
}

.movie-card:hover {
    border: 1px solid #E50914;
    transform: scale(1.02);
}

.explanation {
    color: #BBBBBB;
    font-size: 14px;
    margin-top: 10px;
}

.poster-card {
    background-color: #1F1F1F;
    border-radius: 15px;
    padding: 10px;
    text-align: center;
    transition: 0.3s;
    margin-bottom: 20px;
}

.poster-card:hover {
    transform: scale(1.05);
    border: 2px solid #E50914;
}

.poster-title {
    color: white;
    font-size: 16px;
    font-weight: bold;
    margin-top: 10px;
}

.poster-rating {
    color: #FFD700;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ---------------- #

st.markdown(
    '<div class="title">🎬 Movie Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Hybrid AI-Powered Mood & Movie Recommendation Engine</div>',
    unsafe_allow_html=True
)

# ---------------- INPUT SECTION ---------------- #

col1, col2 = st.columns(2)

with col1:
    movie_list = recommender.data['title'].values

    selected_movie = st.selectbox(
        "🎥 Select a Movie",
        movie_list
    )

with col2:
    mood_options = [
        "happy",
        "sad",
        "excited",
        "romantic",
        "scary",
        "motivational"
    ]

    selected_mood = st.selectbox(
        "😊 Select Your Mood",
        mood_options
    )

# ---------------- BUTTON ---------------- #

if st.button("🚀 Recommend Movies"):

    recommendations = recommender.recommend_by_mood(
        mood=selected_mood,
        base_movie=selected_movie
    )

    st.markdown("---")

    st.subheader("✨ Recommended For You")

    # Recommendation Cards
    
    for movie in recommendations:

        movie_data = search_movie(movie)

        if not movie_data:
            continue

        movie_id = movie_data["id"]

        details = get_movie_details(movie_id)
        credits = get_movie_credits(movie_id)
        providers = get_watch_providers(movie_id)

        poster_url = get_poster_url(
            details.get("poster_path")
        )

        explanation = recommender.explain_recommendation(
            selected_movie,
            movie
        )

        with st.expander(f"🎬 {movie}"):

            col1, col2 = st.columns([1, 2])

            # ---------------- POSTER ---------------- #

            with col1:

                if poster_url:
                    st.image(poster_url)

            # ---------------- DETAILS ---------------- #

            with col2:

                st.subheader(movie)

                st.write(
                    details.get(
                        "overview",
                        "No description available."
                    )
                )

                st.markdown(
                    f"⭐ Rating: {details.get('vote_average', 'N/A')}"
                )

                # Director
                director = "Unknown"

                for crew_member in credits.get("crew", []):

                    if crew_member["job"] == "Director":
                        director = crew_member["name"]
                        break

                st.markdown(f"🎥 Director: {director}")

                # Cast
                cast = credits.get("cast", [])[:5]

                cast_names = [
                    actor["name"]
                    for actor in cast
                ]

                st.markdown(
                    f"👨‍🎤 Cast: {', '.join(cast_names)}"
                )

                # Explanation
                st.info(explanation)

                # ---------------- WATCH PROVIDERS ---------------- #

                st.markdown("### 🌍 Watch Options")

                available_countries = providers.keys()

                if available_countries:

                    for country in available_countries:

                        country_data = providers[country]

                        st.markdown(f"#### 🌎 {country}")

                        # STREAMING
                        flatrate = country_data.get("flatrate", [])

                        if flatrate:

                            stream_names = [
                                p["provider_name"]
                                for p in flatrate
                            ]

                            st.success(
                                f"📺 Stream: {', '.join(stream_names)}"
                            )

                        # RENT
                        rent = country_data.get("rent", [])

                        if rent:

                            rent_names = [
                                p["provider_name"]
                                for p in rent
                            ]

                            st.info(
                                f"💰 Rent: {', '.join(rent_names)}"
                            )

                        # BUY
                        buy = country_data.get("buy", [])

                        if buy:

                            buy_names = [
                                p["provider_name"]
                                for p in buy
                            ]

                            st.warning(
                                f"🛒 Buy: {', '.join(buy_names)}"
                            )

                    # TMDB Watch Link
                    watch_link = get_watch_link(movie_id)

                    st.markdown(
                        f"""
                        <a href="{watch_link}" target="_blank">
                            <button style="
                                background-color:#E50914;
                                color:white;
                                border:none;
                                padding:10px 20px;
                                border-radius:10px;
                                cursor:pointer;
                                font-size:16px;
                            ">
                                🎬 Watch / Stream Online
                            </button>
                        </a>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.warning(
                        "Watch provider information not available."
                    )
import sys
import os

# --------------------------------------------------
# ADD PROJECT ROOT TO PATH
# --------------------------------------------------

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..')
    )
)

# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

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

from src.utils import get_provider_links

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

recommender = MovieRecommender(
    "data/processed/clean_movies.csv"
)

provider_links = get_provider_links()

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

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

.poster-card {
    background-color: #1F1F1F;
    border-radius: 15px;
    padding: 10px;
    text-align: center;
    transition: 0.3s;
    margin-bottom: 20px;
}

.poster-card:hover {
    transform: scale(1.03);
    border: 2px solid #E50914;
}

.poster-title {
    color: white;
    font-size: 18px;
    font-weight: bold;
    margin-top: 10px;
}

.poster-rating {
    color: #FFD700;
    font-size: 14px;
    margin-bottom: 10px;
}

.stButton > button {
    width: 100%;
    background-color: #E50914;
    color: white;
    border-radius: 10px;
    border: none;
    height: 3em;
    font-size: 16px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #ff1e2d;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">🎬 Movie Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Hybrid AI-Powered Mood & Movie Recommendation Engine</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

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

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

if "selected_movie" not in st.session_state:
    st.session_state.selected_movie = None

# --------------------------------------------------
# BUTTON
# --------------------------------------------------

if st.button("🚀 Recommend Movies"):

    with st.spinner("Finding the best movies for you..."):

        st.session_state.selected_movie = selected_movie

        st.session_state.recommendations = (
            recommender.recommend_by_mood(
                mood=selected_mood,
                base_movie=selected_movie
            )
        )

# --------------------------------------------------
# SHOW RECOMMENDATIONS
# --------------------------------------------------

if st.session_state.recommendations:

    recommendations = st.session_state.recommendations

    st.markdown("---")

    st.subheader("✨ Recommended For You")

    cols = st.columns(3)

    for idx, movie in enumerate(recommendations):

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
            st.session_state.selected_movie,
            movie
        )

        with cols[idx % 3]:

            st.markdown(
                '<div class="poster-card">',
                unsafe_allow_html=True
            )

            # ----------------------------------------
            # POSTER
            # ----------------------------------------

            if poster_url:

                st.image(
                    poster_url,
                    width="stretch"
                )

            # ----------------------------------------
            # TITLE
            # ----------------------------------------

            st.markdown(
                f'<div class="poster-title">{movie}</div>',
                unsafe_allow_html=True
            )

            # ----------------------------------------
            # RATING
            # ----------------------------------------

            st.markdown(
                f'<div class="poster-rating">⭐ {details.get("vote_average", "N/A")}</div>',
                unsafe_allow_html=True
            )

            # ----------------------------------------
            # EXPANDABLE DETAILS
            # ----------------------------------------

            with st.expander("📖 View Details"):

                # Overview
                st.write(
                    details.get(
                        "overview",
                        "No description available."
                    )
                )

                # ------------------------------------
                # DIRECTOR
                # ------------------------------------

                director = "Unknown"

                for crew_member in credits.get("crew", []):

                    if crew_member["job"] == "Director":

                        director = crew_member["name"]

                        break

                st.markdown(
                    f"🎥 **Director:** {director}"
                )

                # ------------------------------------
                # CAST
                # ------------------------------------

                cast = credits.get("cast", [])[:5]

                cast_names = [
                    actor["name"]
                    for actor in cast
                ]

                st.markdown(
                    f"👨‍🎤 **Cast:** {', '.join(cast_names)}"
                )

                # ------------------------------------
                # EXPLANATION
                # ------------------------------------

                st.info(explanation)

                # ------------------------------------
                # WATCH OPTIONS
                # ------------------------------------

                st.markdown("### 🌍 Watch Options")

                available_countries = list(providers.keys())

                if available_countries:

                    selected_country = st.selectbox(
                        f"Select Country for {movie}",
                        available_countries,
                        key=f"{movie}_country"
                    )

                    country_data = providers[selected_country]

                    # --------------------------------
                    # STREAM
                    # --------------------------------

                    st.markdown("#### 📺 Stream")

                    flatrate = country_data.get(
                        "flatrate",
                        []
                    )

                    if flatrate:

                        for p in flatrate:

                            provider_name = p["provider_name"]

                            provider_link = provider_links.get(
                                provider_name,
                                "#"
                            )

                            st.markdown(
                                f"""
                                - <a href="{provider_link}" target="_blank">
                                {provider_name}
                                </a>
                                """,
                                unsafe_allow_html=True
                            )

                    else:

                        st.info(
                            "No streaming providers available."
                        )

                    # --------------------------------
                    # RENT
                    # --------------------------------

                    st.markdown("#### 💰 Rent")

                    rent = country_data.get(
                        "rent",
                        []
                    )

                    if rent:

                        rent_names = [
                            p["provider_name"]
                            for p in rent
                        ]

                        st.write(
                            ", ".join(rent_names)
                        )

                    else:

                        st.info(
                            "No rent options available."
                        )

                    # --------------------------------
                    # BUY
                    # --------------------------------

                    st.markdown("#### 🛒 Buy")

                    buy = country_data.get(
                        "buy",
                        []
                    )

                    if buy:

                        buy_names = [
                            p["provider_name"]
                            for p in buy
                        ]

                        st.write(
                            ", ".join(buy_names)
                        )

                    else:

                        st.info(
                            "No buy options available."
                        )

                    # --------------------------------
                    # WATCH LINK
                    # --------------------------------

                    watch_link = get_watch_link(
                        movie_id
                    )

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
                                🎬 View More Watch Options
                            </button>
                        </a>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.warning(
                        "Provider information unavailable."
                    )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )
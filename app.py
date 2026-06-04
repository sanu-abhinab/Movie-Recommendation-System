from flask import (
    Flask,
    render_template,
    request,
    session,
    redirect,
    url_for,
    jsonify
)

from datetime import datetime

from src.tmdb_api import (
    get_popular_movies,
    get_now_playing_movies,
    get_trending_movies,
    get_poster_url,
    search_movie,
    get_movie_details,
    get_movie_credits,
    get_watch_providers,
    get_movie_trailer
)

from src.recommendation import MovieRecommender

app = Flask(__name__)

app.secret_key = "movie_discovery_ai_2026"

# --------------------------------------------------
# LOAD RECOMMENDER
# --------------------------------------------------

recommender = MovieRecommender(
    "data/processed/clean_movies.csv"
)

# --------------------------------------------------
# PROVIDER PRIORITY
# --------------------------------------------------

priority_providers = [

    "Netflix",

    "Amazon Prime Video",

    "Disney Plus",

    "Disney+ Hotstar",

    "Hulu",

    "Max",

    "Apple TV"
]

provider_links = {

    "Netflix":
    "https://www.netflix.com",

    "Amazon Prime Video":
    "https://www.primevideo.com",

    "Disney Plus":
    "https://www.disneyplus.com",

    "Disney+ Hotstar":
    "https://www.hotstar.com",

    "Hulu":
    "https://www.hulu.com",

    "Max":
    "https://www.max.com",

    "Apple TV":
    "https://tv.apple.com"
}




@app.route("/set-engine", methods=["POST"])
def set_engine():

    engine = request.form.get(
        "engine",
        "content"
    )

    session["engine"] = engine

    return redirect(request.referrer or "/")



@app.context_processor
def inject_engine():

    return {

        "selected_engine":

        session.get(
            "engine",
            "content"
        )
    }




# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    popular_movies = get_popular_movies()

    now_playing = get_now_playing_movies()
    
    trending_movies = get_trending_movies()

    return render_template(

        "index.html",
        
        trending_movies=trending_movies,

        popular_movies=popular_movies,

        now_playing=now_playing
        
    )

# --------------------------------------------------
# MOVIE-BASED PAGE
# --------------------------------------------------

@app.route("/movie-based")
def movie_based():

    movie_list = recommender.data['title'].values

    return render_template(

        "movie_based.html",

        movie_list=movie_list
    )


# --------------------------------------------------
# MOVIE RECOMMENDATIONS
# --------------------------------------------------

@app.route("/recommend", methods=["POST"])
def recommend():

    movie_name = request.form.get("movie")

    recommendations = recommender.recommend(
        movie_name
    )[:8]
    
    recommendations = recommender.recommend(movie_name)

    movie_data = []

    for movie in recommendations:

        # ------------------------------------------
        # SEARCH MOVIE
        # ------------------------------------------

        search = search_movie(movie)

        if not search:
            continue

        movie_id = search["id"]

        # ------------------------------------------
        # DETAILS
        # ------------------------------------------

        details = get_movie_details(
            movie_id
        )
        
        trailer_link = get_movie_trailer(
            movie_id
        )

        poster = get_poster_url(
            details.get("poster_path")
        )

        explanation = recommender.explain_recommendation(
            movie_name,
            movie
        )
        
        credits = get_movie_credits(movie_id)
        director = "Unknown"
        
        for crew in credits.get("crew", []):
            if crew.get("job") == "Director":
                director = crew["name"]
                break
            
        cast = ", ".join([
            actor["name"]
            for actor in credits.get(
                "cast",
                []
            )[:5]
        ])
        
        production_company = "Unknown"
        
        if details.get(
            "production_companies"
        ):
            
            production_company = details[
                "production_companies"
            ][0]["name"]

        # ------------------------------------------
        # PROVIDERS
        # ------------------------------------------

        providers = get_watch_providers(
            movie_id
        )

        provider_name = None

        provider_link = None

        for country in providers.values():

            flatrate = country.get(
                "flatrate",
                []
            )

            for p in flatrate:

                name = p["provider_name"]

                if name in priority_providers:

                    provider_name = name

                    provider_link = provider_links.get(
                        name
                    )

                    break

            if provider_name:
                break

        # ------------------------------------------
        # APPEND MOVIE
        # ------------------------------------------
        
        

        movie_data.append({

            "title": movie,

            "poster": poster,
            
            "backdrop":
            f"https://image.tmdb.org/t/p/original{details.get('backdrop_path')}"
            if details.get("backdrop_path")
            else None,
            

            "rating": details.get(
                "vote_average",
                "N/A"
            ),

            "overview": details.get(
                "overview",
                "No overview available."
            ),

            "runtime": details.get(
                "runtime",
                "N/A"
            ),

            "genres": [

                genre["name"]

                for genre in details.get(
                    "genres",
                    []
                )
            ],

            "explanation": explanation,

            "movie_id": movie_id,

            "provider_name":
            provider_name,

            "provider_link":
            provider_link,
            
            "release_date":
            details.get(
                "release_date",
                "N/A"
            ),
            
            "director":
            director,
            
            "cast":
            cast,
            
            "trailer_link":
            trailer_link,
            
            "production_company":
            production_company
            
        })

    return render_template(

        "recommendations.html",

        selected_movie=movie_name,

        recommendations=movie_data
    )

# --------------------------------------------------
# MOOD-BASED PAGE
# --------------------------------------------------

@app.route("/mood-based")
def mood_based():
    
    return render_template(
        "mood_based.html"
    )

# --------------------------------------------------
# WATCH CONTEXT PAGE
# --------------------------------------------------

@app.route("/watch-context", methods=["POST"])
def watch_context():

    mood = request.form.get("mood")

    return render_template(

        "watch_context.html",

        mood=mood
    )

# --------------------------------------------------
# GENRE PAGE
# --------------------------------------------------

@app.route("/genre", methods=["POST"])
def genre():

    mood = request.form.get("mood")

    context = request.form.get("context")

    genres = [

        "Action",

        "Adventure",

        "Animation",

        "Comedy",

        "Crime",

        "Drama",

        "Fantasy",

        "Horror",

        "Mystery",

        "Romance",

        "Science Fiction",

        "Thriller",

        "Doesn't Matter"
    ]

    return render_template(

        "genre.html",

        mood=mood,

        context=context,

        genres=genres
    )

# --------------------------------------------------
# PREFERENCE PAGE
# --------------------------------------------------

@app.route("/preference", methods=["POST"])
def preference():

    mood = request.form.get("mood")

    context = request.form.get("context")

    genre = request.form.get("genre")

    return render_template(

        "preference.html",

        mood=mood,

        context=context,

        genre=genre
    )

# --------------------------------------------------
# MOOD RECOMMENDATIONS
# --------------------------------------------------

@app.route("/mood-recommendations", methods=["POST"])
def mood_recommendations():

    mood = request.form.get("mood")

    context = request.form.get("context")

    genre = request.form.get("genre")

    preference = request.form.get(
        "preference"
    )

    # ------------------------------------------
    # GENRE PRIORITY SYSTEM
    # ------------------------------------------

    if genre != "Doesn't Matter":

        filtered_movies = recommender.data[

            recommender.data["genres"].str.contains(
                genre,
                case=False,
                na=False
            )
        ]

        recommendations = filtered_movies[
            "title"
        ].head(30).tolist()

    else:

        recommendations = recommender.recommend_by_mood(
            mood=mood
        )

    movie_data = []

    fallback_movies = []

    current_year = datetime.now().year

    for movie in recommendations:

        # ------------------------------------------
        # SEARCH MOVIE
        # ------------------------------------------

        search = search_movie(movie)

        if not search:
            continue

        movie_id = search["id"]

        # ------------------------------------------
        # DETAILS
        # ------------------------------------------

        details = get_movie_details(
            movie_id
        )

        poster = get_poster_url(
            details.get("poster_path")
        )

        # ------------------------------------------
        # GENRES
        # ------------------------------------------

        movie_genres = [

            g["name"]

            for g in details.get(
                "genres",
                []
            )
        ]

        # ------------------------------------------
        # RELEASE DATE
        # ------------------------------------------

        release_date = details.get(
            "release_date",
            ""
        )

        release_year = None

        if release_date:

            try:

                release_year = int(
                    release_date[:4]
                )

            except:

                release_year = None
                
        credits = get_movie_credits(movie_id)
        director = "Unknown"
        
        for crew in credits.get("crew", []):
            if crew.get("job") == "Director":
                director = crew["name"]
                break
            
        cast = ", ".join([
            actor["name"]
            for actor in credits.get(
                "cast",
                []
            )[:5]
        ])
        
        production_company = "Unknown"
        
        if details.get(
            "production_companies"
        ):
            
            production_company = details[
                "production_companies"
            ][0]["name"]

        # ------------------------------------------
        # PROVIDERS
        # ------------------------------------------

        providers = get_watch_providers(
            movie_id
        )

        provider_name = None

        provider_link = None

        for country in providers.values():

            flatrate = country.get(
                "flatrate",
                []
            )

            for p in flatrate:

                name = p["provider_name"]

                if name in priority_providers:

                    provider_name = name

                    provider_link = provider_links.get(
                        name
                    )

                    break

            if provider_name:
                break
            
        trailer_link = get_movie_trailer(
            movie_id
        )

        # ------------------------------------------
        # FALLBACK MOVIES
        # ------------------------------------------

        fallback_movies.append({

            "title": movie,

            "poster": poster,

            "rating": details.get(
                "vote_average",
                "N/A"
            ),

            "overview": details.get(
                "overview",
                "No overview available."
            ),

            "runtime": details.get(
                "runtime",
                "N/A"
            ),

            "genres": movie_genres,

            "explanation":

            f"Recommended for your "
            f"{mood} mood.",

            "provider_name":
            provider_name,

            "provider_link":
            provider_link,
            
            "release_date":
            details.get(
                "release_date",
                "N/A"
            ),
            
            "director":
            director,
            
            "cast":
            cast,
            
            "production_company":
            production_company,
            
            "trailer_link":
            trailer_link,
        })

        # ------------------------------------------
        # NEW GEN FILTER
        # ------------------------------------------

        if preference == "new":

            if release_year:

                if release_year < current_year - 5:

                    continue

        # ------------------------------------------
        # APPEND FILTERED MOVIE
        # ------------------------------------------

        movie_data.append({

            "title": movie,

            "poster": poster,
            
            "backdrop":
            f"https://image.tmdb.org/t/p/original{details.get('backdrop_path')}"
            if details.get('backdrop_path')
            else None,

            "rating": details.get(
                "vote_average",
                "N/A"
            ),

            "overview": details.get(
                "overview",
                "No overview available."
            ),

            "runtime": details.get(
                "runtime",
                "N/A"
            ),

            "genres": movie_genres,

            "explanation":

            f"Recommended because it matches your "
            f"{genre} preference and "
            f"{mood} mood while watching with "
            f"{context}.",

            "provider_name":
            provider_name,

            "provider_link":
            provider_link,
            
            "release_date":
            details.get(
                "release_date",
                "N/A"
            ),
            
            "director":
            director,
            
            "cast":
            cast,
            
            "trailer_link":
            trailer_link,
            
            "production_company":
            production_company
        })

    # ------------------------------------------
    # FALLBACK SYSTEM
    # ------------------------------------------

    fallback_message = None

    if len(movie_data) == 0:

        movie_data = fallback_movies[:8]

        fallback_message = (
            "No exact matches found. "
            "Showing closest recommendations instead."
        )

    # Limit final results
    movie_data = movie_data[:8]
    
    return render_template(

        "recommendations.html",

        selected_movie=
        f"{mood.title()} Mood",

        recommendations=movie_data,

        fallback_message=fallback_message
    )


# --------------------------------------------------
# MOVIE DETAILS API
# --------------------------------------------------

@app.route("/movie-details/<int:movie_id>")
def movie_details(movie_id):

    try:

        details = get_movie_details(
            movie_id
        )

        credits = get_movie_credits(
            movie_id
        )

        # ------------------------------------------
        # CAST
        # ------------------------------------------

        cast = []

        for actor in credits.get(
            "cast",
            []
        )[:5]:

            cast.append(
                actor["name"]
            )

        # ------------------------------------------
        # DIRECTOR
        # ------------------------------------------

        director = "Unknown"

        for crew in credits.get(
            "crew",
            []
        ):

            if crew.get(
                "job"
            ) == "Director":

                director = crew["name"]

                break

        # ------------------------------------------
        # PRODUCTION COMPANY
        # ------------------------------------------

        production_company = "Unknown"

        companies = details.get(
            "production_companies",
            []
        )

        if companies:

            production_company = companies[
                0
            ]["name"]

        # ------------------------------------------
        # STREAMING PROVIDERS
        # ------------------------------------------

        providers = get_watch_providers(
            movie_id
        )

        provider_name = None

        provider_link = None

        for country in providers.values():

            flatrate = country.get(
                "flatrate",
                []
            )

            for p in flatrate:

                name = p["provider_name"]

                if name in priority_providers:

                    provider_name = name

                    provider_link = provider_links.get(
                        name
                    )

                    break

            if provider_name:
                break

        # ------------------------------------------
        # IMDb STYLE RATING
        # ------------------------------------------

        imdb_rating = round(

            details.get(
                "vote_average",
                0
            ),

            1
        )

        # ------------------------------------------
        # RETURN JSON
        # ------------------------------------------

        return jsonify({

            "title":
            details.get("title"),

            "overview":
            details.get("overview"),

            "poster":
            get_poster_url(
                details.get(
                    "poster_path"
                )
            ),
            
            "backdrop":
            f"https://image.tmdb.org/t/p/original{details.get('backdrop_path')}"
            if details.get('backdrop_path')
            else None,

            "rating":
            details.get(
                "vote_average"
            ),

            "imdb_rating":
            imdb_rating,

            "release_date":
            details.get(
                "release_date"
            ),

            "runtime":
            details.get(
                "runtime"
            ),

            "genres": [

                genre["name"]

                for genre in details.get(
                    "genres",
                    []
                )
            ],

            "director":
            director,

            "cast":
            cast,

            "production_company":
            production_company,

            "provider":
            provider_name,

            "provider_link":
            provider_link
            
        })

    except Exception as e:

        return jsonify({

            "error":
            str(e)

        }), 500

# --------------------------------------------------
# RUN APP
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=True
    )
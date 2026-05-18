import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 🔥 TMDB Bearer Token
ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIwYjk2MzA4NjNmMDA1MjM1Mzc3Y2VmNDdkYjFiZTc5ZCIsIm5iZiI6MTc3NTYzNDI1NS4xMTYsInN1YiI6IjY5ZDYwNzRmNzI0NTIwYjNkZDA3MTk1ZCIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.UeIm-BLGRSq0ESjUsjhkxv9ldu1rBkIpDguIh6K2qPc"

BASE_URL = "https://api.themoviedb.org/3"

# Headers
headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "accept": "application/json"
}

# Retry Strategy
session = requests.Session()

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504]
)

adapter = HTTPAdapter(max_retries=retry_strategy)

session.mount("https://", adapter)
session.mount("http://", adapter)


# --------------------------------------------
# GET POPULAR MOVIES
# --------------------------------------------

def get_popular_movies():

    url = f"{BASE_URL}/movie/popular"

    try:
        response = session.get(
            url,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        movies = []

        for movie in data.get("results", []):
            movies.append(movie["title"])

        return movies

    except requests.exceptions.RequestException as e:
        return [f"TMDB Connection Error: {e}"]


# --------------------------------------------
# GET MOVIE DETAILS
# --------------------------------------------

def get_movie_details(movie_id):

    url = f"{BASE_URL}/movie/{movie_id}"

    try:
        response = session.get(
            url,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        return {
            "error": str(e)
        }


# --------------------------------------------
# SEARCH MOVIE
# --------------------------------------------

def search_movie(movie_name):

    url = f"{BASE_URL}/search/movie"

    params = {
        "query": movie_name
    }

    try:
        response = session.get(
            url,
            headers=headers,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        if data["results"]:
            return data["results"][0]

        return None

    except requests.exceptions.RequestException as e:
        return {
            "error": str(e)
        }

# GET MOVIE CREDITS
        
def get_movie_credits(movie_id):

    url = f"{BASE_URL}/movie/{movie_id}/credits"

    try:
        response = session.get(
            url,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        return {
            "error": str(e)
        }
        
# GET WATCH PROVIDERS

def get_watch_providers(movie_id):

    url = f"{BASE_URL}/movie/{movie_id}/watch/providers"

    try:
        response = session.get(
            url,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        return data.get("results", {})

    except requests.exceptions.RequestException as e:
        return {
            "error": str(e)
        }

# GET POSTER URL

def get_poster_url(poster_path):

    if not poster_path:
        return None

    return f"https://image.tmdb.org/t/p/w500{poster_path}"

# GET WATCH LINKS

def get_watch_link(movie_id):

    return f"https://www.themoviedb.org/movie/{movie_id}/watch"
import requests

API_KEY = "0b9630863f005235377cef47db1be79d"


def get_popular_movies():
    url = "https://api.themoviedb.org/3/movie/popular"

    params = {
        "api_key": API_KEY,
        "language": "en-US",
        "page": 1
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()

        movies = [movie['title'] for movie in data.get('results', [])]
        return movies

    except Exception as e:
        return [f"Error fetching data: {e}"]


def get_now_playing_movies():
    url = "https://api.themoviedb.org/3/movie/now_playing"

    params = {
        "api_key": API_KEY,
        "language": "en-US",
        "page": 1
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()

        movies = [movie['title'] for movie in data.get('results', [])]
        return movies

    except Exception as e:
        return [f"Error fetching data: {e}"]
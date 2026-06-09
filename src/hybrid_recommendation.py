import pandas as pd

from src.recommendation import MovieRecommender


class HybridRecommender(MovieRecommender):

    def __init__(self, csv_path):

        super().__init__(csv_path)
        
    def recommend(
        self,
        movie_title,
        top_n=8
    ):
        if movie_title not in self.data['title'].values:

            return ["Movie not found"]

        # ----------------------------------
        # FIND MOVIE INDEX
        # ----------------------------------

        index = self.data[
            self.data['title'] == movie_title
        ].index[0]

        # ----------------------------------
        # GET MOVIE DATA
        # ----------------------------------

        target_movie = self.data.iloc[index]

        target_genres = set(
            eval(target_movie['genres'])
        )

        target_cast = set(
            eval(target_movie['cast'])
        )

        target_director = target_movie['crew']

        # ----------------------------------
        # COSINE SIMILARITY
        # ----------------------------------

        similarity_scores = list(
            enumerate(
                self.similarity_matrix[index]
            )
        )

        hybrid_scores = []

        for movie_index, similarity in similarity_scores:

            if movie_index == index:

                continue

            movie = self.data.iloc[movie_index]

            score = similarity

            # ----------------------------------
            # GENRE BOOST
            # ----------------------------------

            genres = set(
                eval(movie['genres'])
            )

            shared_genres = len(
                target_genres.intersection(
                    genres
                )
            )

            score += (
                shared_genres * 0.05
            )

            # ----------------------------------
            # CAST BOOST
            # ----------------------------------

            cast = set(
                eval(movie['cast'])
            )

            shared_cast = len(
                target_cast.intersection(
                    cast
                )
            )

            score += (
                shared_cast * 0.03
            )

            # ----------------------------------
            # DIRECTOR BOOST
            # ----------------------------------

            if movie['crew'] == target_director:

                score += 0.10

            hybrid_scores.append(

                (
                    movie_index,
                    score
                )

            )

        # ----------------------------------
        # SORT BY HYBRID SCORE
        # ----------------------------------

        hybrid_scores = sorted(

            hybrid_scores,

            key=lambda x: x[1],

            reverse=True

        )

        recommendations = []

        for movie_index, score in hybrid_scores[:top_n]:

            recommendations.append(

                self.data.iloc[
                    movie_index
                ].title

            )

        return recommendations
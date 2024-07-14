import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

file_path_ratings = 'ratings.csv'
file_path_movies = 'movies.csv'

df_ratings = pd.read_csv(file_path_ratings)
df_movies = pd.read_csv(file_path_movies)


def recommend_movies(user_id, num_recommendations=10):
    user_row_number = user_id - 1

    user_predictions = preds_df.iloc[user_row_number]

    already_rated = df_ratings[df_ratings['userId'] == user_id]
    already_rated_full = already_rated.merge(df_movies, how='left', on='movieId')[
        ['title', 'genres', 'rating']
    ]

    sorted_user_predictions = user_predictions.sort_values(ascending=False)

    recommendations = df_movies[~df_movies['movieId'].isin(already_rated['movieId'])]
    recommendations = recommendations.merge(
        pd.DataFrame(sorted_user_predictions).reset_index(), how='left', on='movieId'
    )
    recommendations = recommendations.head(num_recommendations)
    recommendations = recommendations[['title', 'genres']]

    return already_rated_full, recommendations


ratings_matrix = df_ratings.pivot(index='userId', columns='movieId', values='rating')

# Видалення користувачів, які оцінили менше 200 фільмів, та фільмів з менше ніж 100 оцінками
min_user_ratings = 50
min_movie_ratings = 20
ratings_matrix = ratings_matrix.dropna(thresh=min_user_ratings, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=min_movie_ratings, axis=1)

# Заповнення відсутніх значень середнім значенням
ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values


user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

k_max = min(R_demeaned.shape)  # Обчислення максимального значення k

for k in range(1, k_max + 1):
    try:
        U, sigma, Vt = svds(R_demeaned, k=k)
        print(f"SVD успішно виконано з k = {k}")
        break
    except ValueError as e:
        print(f"Помилка SVD: {e}")
        if k == k_max:
            print("Не успішне SVD.")

if U is not None:
    all_user_predicted_ratings = np.dot(np.dot(U, np.diag(sigma)), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)

user_id = 1
num_recommendations = 10

already_rated, recommendations = recommend_movies(user_id, num_recommendations)

print("Вже оцінені фільми користувачем:")
print(already_rated[['title', 'genres', 'rating']])

print("\nРекомендовані фільми:")
print(recommendations[['title', 'genres']])
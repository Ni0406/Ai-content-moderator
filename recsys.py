import os
import urllib.request
import zipfile
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- АРХИТЕКТУРА TWO-TOWER (ДВЕ БАШНИ) ---
class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=32):
        super().__init__()
        # Башня 1: Пользовательская (учит предпочтения человека)
        self.user_tower = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
        
        # Башня 2: Контентная (учит свойства фильма)
        self.movie_tower = nn.Embedding(num_embeddings=num_movies, embedding_dim=embedding_dim)

    def forward(self, user_id, movie_id):
        # Получаем векторы из обеих башен
        user_embedding = self.user_tower(user_id)
        movie_embedding = self.movie_tower(movie_id)
        
        # Перемножаем векторы (dot product), чтобы получить вероятность клика/лайка
        return (user_embedding * movie_embedding).sum(dim=1)


class RealMovieRecSys:
    def __init__(self):
        print("[INFO] Инициализация Рекомендательной Системы...")
        self.data_dir = "ml-100k"
        self.download_and_extract_data()
        self.load_data()

    def download_and_extract_data(self):
        url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
        zip_path = "ml-100k.zip"
        if not os.path.exists(self.data_dir):
            print(f"[INFO] Скачиваю реальный датасет MovieLens 100K...")
            urllib.request.urlretrieve(url, zip_path)
            print("[INFO] Распаковка архива...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove(zip_path)

    def load_data(self):
        ratings_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
        self.ratings = pd.read_csv(f'{self.data_dir}/u.data', sep='\t', names=ratings_cols, encoding='latin-1')

        movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url', 
                       'unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 
                       'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
                       'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.movies = pd.read_csv(f'{self.data_dir}/u.item', sep='|', names=movies_cols, encoding='latin-1')
        
        genres_list = movies_cols[5:]
        def get_genres(row):
            return ' '.join([genre for genre in genres_list if row[genre] == 1])
        self.movies['genres'] = self.movies.apply(get_genres, axis=1)

    def get_popular_heuristic(self, top_n=5):
        print("\n--- 1. Эвристика: САМЫЕ ПОПУЛЯРНЫЕ ФИЛЬМЫ ---")
        movie_stats = self.ratings.groupby('movie_id').agg({'rating': ['mean', 'count']})
        movie_stats.columns = ['avg_rating', 'rating_count']
        popular = movie_stats[movie_stats['rating_count'] > 100].reset_index()
        popular = pd.merge(popular, self.movies[['movie_id', 'title']], on='movie_id')
        popular = popular.sort_values(by='avg_rating', ascending=False).head(top_n)
        for _, row in popular.iterrows():
            print(f"Фильм: {row['title']} | Средняя оценка: {row['avg_rating']:.2f}")

    def content_based_recommendation(self, target_movie_title, top_n=3):
        print(f"\n--- 2. Content-Based: ПОХОЖЕЕ НА '{target_movie_title}' ---")
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(self.movies['genres'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        try:
            idx = self.movies.index[self.movies['title'].str.contains(target_movie_title, case=False, na=False)].tolist()[0]
        except IndexError:
            return
            
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_indices = [i[0] for i in sim_scores[1:top_n+1]]
        for i in top_indices:
            print(f"Рекомендуем: {self.movies['title'].iloc[i]}")

    def train_two_tower_model(self, epochs=3):
        """СОВРЕМЕННЫЙ ПОДХОД: Обучение нейросети Two-Tower на PyTorch"""
        print("\n--- 3. Продвинутая архитектура: Обучение TWO-TOWER нейросети ---")
        
        # Подготовка данных для PyTorch (приводим ID к формату от 0 до N)
        users = torch.tensor(self.ratings['user_id'].values - 1, dtype=torch.long)
        movies = torch.tensor(self.ratings['movie_id'].values - 1, dtype=torch.long)
        
        # Бинарная классификация: рейтинг >= 4 это "лайк" (1), иначе "дизлайк" (0)
        ratings = torch.tensor((self.ratings['rating'].values >= 4).astype(float), dtype=torch.float32)

        # Инициализация модели, функции потерь и оптимизатора
        num_users = self.ratings['user_id'].max()
        num_movies = self.ratings['movie_id'].max()
        
        model = TwoTowerModel(num_users, num_movies)
        criterion = nn.BCEWithLogitsLoss() # Функция потерь для бинарной классификации
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Мини-цикл обучения
        print(f"[INFO] Начинаем обучение ({epochs} эпох) на {len(ratings)} оценках...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass (пропускаем данные через башни)
            predictions = model(users, movies)
            
            # Вычисляем ошибку
            loss = criterion(predictions, ratings)
            
            # Backward pass (обновляем веса нейросети)
            loss.backward()
            optimizer.step()
            
            print(f"Эпоха {epoch+1}/{epochs} | Ошибка (Loss): {loss.item():.4f}")
            
        print("[INFO] Модель Two-Tower успешно обучена!")

if __name__ == "__main__":
    recsys = RealMovieRecSys()
    
    # 1. Запуск эвристики
    recsys.get_popular_heuristic()
    
    # 2. Запуск Content-Based
    recsys.content_based_recommendation("Star Wars")
    
    # 3. Обучение современной нейросети (Two-Tower)
    recsys.train_two_tower_model()
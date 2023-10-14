# %% [markdown]
# # Laporan Proyek Machine Learning - Submission 2: System Recommender
# - Nama: Imam Agus Faisal
# - Email: imamagusfaisal120@gmail.com
# - Id Dicoding: imamaf

# %% [markdown]
# ## Problem Statement

# %% [markdown]
# - Bagaimana cara membuat rekomendasi film yang mempunyai karakteristik hampir sama dengan film yang pernah dilihat?
# - Bagaimana cara merekomendasikan film berdasarkan preferensi pengguna?

# %% [markdown]
# ## Menyiapkan semua library yang dibutuhkan

# %%
import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

# %% [markdown]
# ## Data Wrangling

# %% [markdown]
# **Memuat setiap tabel pada dataset**

# %%
movies_df=pd.read_csv("datasets/movies.csv")
movies_df.head()

# %%
ratings_df=pd.read_csv("datasets/ratings.csv")
ratings_df.head()

# %% [markdown]
# ### Assessing Data
# **Melihat informasi, Memeriksa missing value, Memeriksa duplikasi, dan Memeriksa parameter statistik pada setiap tabel**

# %%
print('\n', movies_df.info())
print('\nMissing value movies:\n', movies_df.isnull().sum())
print('\nJumlah duplikasi movies:\n', movies_df.duplicated().sum())
print('\n\nParameter statistik movies:\n', movies_df.describe())

# %%
movies_df.head()

# %%
print('\n', ratings_df.info())
print('\nMissing value ratings:\n', ratings_df.isnull().sum())
print('\nJumlah duplikasi ratings:\n', ratings_df.duplicated().sum())
print('\n\nParameter statistik ratings:\n', ratings_df.describe())

# %%
ratings_df.head()

# %% [markdown]
# **Rangkuman Hasil Analisis Tahap Assesing Data pada Dataset:**
# *   Tidak terdapat missing value
# *   Tidak terdapat adanya duplikasi
# *   Terdapat innacurate type value pada kolom timestamp
# *   Terdapat tahun movie diproduksi yang dapat dipindahkan ke dalam kolom tersendiri
# 
# 

# %% [markdown]
# ### Cleaning Data
# Memisahkan tahun produksi movie kedalam kolom tersendiri

# %%
#Using regular expressions to find a year stored between parentheses

#We specify the parantheses so we don't conflict with movies that have years in their titles
movies_df['year'] = movies_df.title.str.extract(r'\((\d{4})\)')

#Removing the years from the 'title' column
movies_df['title'] = movies_df.title.str.replace(r'\((\d{4})\)', '')

#Applying the strip function to get rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

# %% [markdown]
# Mengubah timestamp bertipe int64 menjadi datetime

# %%
ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'], unit='s')

# %% [markdown]
# Mengecek kembali dataset yang sudah dibersihkan

# %%
movies_df.head()

# %%
ratings_df.head()

# %% [markdown]
# ## Exploratory Data Analysis (EDA)

# %% [markdown]
# ### Univariate Analysis

# %% [markdown]
# Jumlah movie yang dirilis setiap tahun

# %%
# Group movies by year and count the number of movies
movie_counts = movies_df['year'].value_counts().sort_index()

# Plotting the number of movies released each year
plt.figure(figsize=(10, 6))
plt.bar(movie_counts.index, movie_counts.values)
plt.xlabel('Tahun')
plt.xticks(fontsize=5, rotation=90)
plt.ylabel('Jumlah Movie')
plt.title('Jumlah Movie yang Dirilis Setiap Tahun')
plt.show()

# %% [markdown]
# Jumlah rating yang diberikan setiap tahun

# %%
rating_counts = ratings_df['timestamp'].dt.year.value_counts().sort_index()

plt.figure(figsize=(10, 6))
plt.bar(rating_counts.index, rating_counts.values)
plt.xlabel('Tahun')
plt.ylabel('Jumlah Rating')
plt.title('Jumlah Rating Setiap Tahun')
plt.show()

# %% [markdown]
# ### Multivariate Analysis

# %% [markdown]
# Gabungkan dataset movie dengan datset rating

# %%
merged_df = movies_df.merge(ratings_df, on='movieId')

# %% [markdown]
# Top 10 Movie dengan jumlah rating terbanyak

# %%
# Count the number of ratings for each movie
movie_ratings = merged_df['title'].value_counts().head(10)

# Plotting the top 10 most rated movies
plt.figure(figsize=(10, 6))
sns.barplot(x=movie_ratings.values, y=movie_ratings.index)
plt.xlabel('Jumlah Rating')
plt.ylabel('Judul Movie')
plt.title('Top 10 Movie dengan Jumlah Rating Terbanyak')
plt.show()

# %% [markdown]
# Top 10 Movie dengan jumlah rating paling sedikit

# %%
# Count the number of ratings for each movie
movie_ratings = merged_df['title'].value_counts().tail(10)

# Plotting the top 10 most rated movies
plt.figure(figsize=(10, 6))
sns.barplot(x=movie_ratings.values, y=movie_ratings.index)
plt.xlabel('Jumlah Rating')
plt.ylabel('Judul Movie')
plt.title('Top 10 Movie dengan Jumlah Rating Paling Sedikit')
plt.show()

# %% [markdown]
# Distribusi Rating pada Semua Movie

# %%
plt.figure(figsize=(10, 6))
sns.histplot(ratings_df['rating'])
plt.xlabel('Rating')
plt.ylabel('Frekuensi')
plt.title('Distribusi Rating pada Semua Movie')
plt.grid()
plt.show()

# %% [markdown]
# Distribusi Rating pada Movie Tertentu

# %%
movie_id = 1  # Specify the movie ID for analysis

# Filter ratings data for the specific movie
movie_ratings = ratings_df[ratings_df['movieId'] == movie_id]['rating']

# Plotting the distribution of ratings for the specific movie
plt.figure(figsize=(10, 6))
sns.histplot(movie_ratings, kde=True)
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.title(f'Distribution of Ratings for Movie {movie_id}')
plt.grid()
plt.show()

# %% [markdown]
# Rata-rata Rating Setiap Tahun

# %%
# Calculate the average rating for each year
avg_ratings = merged_df.groupby('year')['rating'].mean()

# Plotting the average rating for each year
plt.figure(figsize=(10, 6))
plt.plot(avg_ratings.index, avg_ratings.values, marker='o')
plt.xlabel('Tahun')
plt.xticks(fontsize=5, rotation=90)
plt.ylabel('Rata-rata Rating')
plt.title('Rata-rata Rating Setiap Tahun')
plt.grid()
plt.show()

# %% [markdown]
# Rata-rata Rating Setiap Movie

# %%
# Average rating for each movie genre
average_rating_movie = merged_df.groupby('movieId')['rating'].mean()

print(average_rating_movie)

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# ### Data Preparation untuk Content Based Filtering

# %% [markdown]
# Mengurutkan movie berdasarkan Id

# %%
prep_movies = merged_df.sort_values(by='movieId')

# %% [markdown]
# Mengecek jumlah unique movie berdasarkan Id

# %%
len(prep_movies.movieId.unique())

# %% [markdown]
# Menghilangkan duplikat movieId

# %%
prep_movies = prep_movies.drop_duplicates('movieId')

# %% [markdown]
# Mengkonversi data menjadi list

# %%
# Mengonversi data series ‘movieId’ menjadi dalam bentuk list
movie_id = prep_movies['movieId'].tolist()
 
# Mengonversi data series ‘title’ menjadi dalam bentuk list
movie_name = prep_movies['title'].tolist()
 
# Mengonversi data series ‘genres’ menjadi dalam bentuk list
movie_genre = prep_movies['genres'].tolist()

# %% [markdown]
# Mengecek panjang kolom

# %%
print(len(movie_id))
print(len(movie_name))
print(len(movie_genre))

# %% [markdown]
# Membuat dictionary untuk menentukan pasangan Key-Value

# %%
movie_dict = pd.DataFrame({
    'id':movie_id,
    'movie_name':movie_name,
    'genre':movie_genre
})
movie_dict

# %% [markdown]
# ### Data Preparation untuk Collaborative Filtering

# %% [markdown]
# Membuat dataframe untuk collaborative filtering

# %%
collaborative_df = ratings_df

# %% [markdown]
# Encoding userId dan MovieId ke dalam indeks integer

# %%
# Mengubah userId menjadi list tanpa nilai yang sama
user_ids = collaborative_df['userId'].unique().tolist()
print('list userID: ', user_ids)
 
# Melakukan encoding userId
user_to_user_encoded = {x: i for i, x in enumerate(user_ids)}
print('encoded userID : ', user_to_user_encoded)
 
# Melakukan proses encoding angka ke userId
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
print('encoded angka ke userID: ', user_encoded_to_user)

# %%
# Mengubah movieId menjadi list tanpa nilai yang sama
movie_ids = collaborative_df['movieId'].unique().tolist()
print('list movieId: ', movie_ids)
 
# Melakukan encoding movieId
movie_to_movie_encoded = {x: i for i, x in enumerate(movie_ids)}
print('encoded movieId : ', movie_to_movie_encoded)
 
# Melakukan proses encoding angka ke movieId
movie_encoded_to_movie = {i: x for i, x in enumerate(movie_ids)}
print('encoded angka ke movieId: ', movie_encoded_to_movie)

# %% [markdown]
# Memetakan userId dan MovieId ke dalam dataframe

# %%
# Mapping userId ke dataframe user
collaborative_df['user'] = collaborative_df['userId'].map(user_to_user_encoded)
 
# Mapping movieId ke dataframe movie
collaborative_df['movie'] = collaborative_df['movieId'].map(movie_to_movie_encoded)

# %% [markdown]
# Mengecek jumlah data user dan movie

# %%
# Mendapatkan jumlah user
num_users = len(user_to_user_encoded)
 
# Mendapatkan jumlah movie
num_movie = len(movie_encoded_to_movie)
 
# Nilai minimum rating
min_rating = min(collaborative_df['rating'])
 
# Nilai maksimal rating
max_rating = max(collaborative_df['rating'])
 
print('Number of User: {}, Number of Movie: {}, Min Rating: {}, Max Rating: {}'.format(
    num_users, num_movie, min_rating, max_rating
))

# %% [markdown]
# ##### Membagi data untuk Training dan Validasi

# %% [markdown]
# Mengacak data

# %%
collaborative_df = collaborative_df.sample(frac=1, random_state=42)
collaborative_df

# %% [markdown]
# Memetakan data user dan movie, lalu membagi data training dan validasi dengan rasio 80:20

# %%
# Membuat variabel x untuk mencocokkan data user dan movie menjadi satu value
x = collaborative_df[['user', 'movie']].values
 
# Membuat variabel y untuk membuat rating dari hasil 
y = collaborative_df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
 
# Membagi menjadi 80% data train dan 20% data validasi
train_indices = int(0.8 * collaborative_df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)

print(x, y)

# %% [markdown]
# ## Model Development

# %% [markdown]
# ### Content Based Filtering

# %% [markdown]
# TF-IDF Vectorizer

# %%
# Inisialisasi TfidfVectorizer
tfid = TfidfVectorizer()
 
# Melakukan perhitungan idf pada data genre
tfid.fit(movie_dict['genre']) 
 
# Mapping array dari fitur index integer ke fitur nama
tfid.get_feature_names_out() 

# %% [markdown]
# Melakukan Fit dan Transformasikan ke bentuk Matrik

# %%
tfidf_matrix = tfid.fit_transform(movie_dict['genre']) 
tfidf_matrix.todense()

# %% [markdown]
# Menghitung kesamaan dengan **Cosine Similarity**

# %%
movie_cosine = cosine_similarity(tfidf_matrix)
movie_cosine

# %% [markdown]
# Membuat dataframe dari hasil Cosine Similarity

# %%
movie_cosine_df = pd.DataFrame(movie_cosine, index=movie_dict['movie_name'], columns=movie_dict['movie_name'])
print('Shape:', movie_cosine_df.shape)
 
movie_cosine_df.sample(5, axis=1).sample(10, axis=0)

# %% [markdown]
# membuat fungsi movie_recommendations dengan beberapa parameter sebagai berikut:
# - movie_name : Nama judul dari movie tersebut (index kemiripan dataframe).
# - similarity_data : Dataframe mengenai similarity yang telah kita didefinisikan sebelumnya
# - items : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini adalah ‘movie_name’ dan ‘genre’.
# - k : Banyak rekomendasi yang ingin diberikan.

# %%
def movie_recommendations(movie_name, similarity=movie_cosine_df, items=movie_dict[['movie_name', 'genre']], k=5):
   
 
    # Mengambil data dengan menggunakan argpartition untuk melakukan partisi secara tidak langsung sepanjang sumbu yang diberikan    
    # Dataframe diubah menjadi numpy
    # Range(start, stop, step)
    index = similarity.loc[:,movie_name].to_numpy().argpartition(
        range(-1, -k, -1))
    
    # Mengambil data dengan similarity terbesar dari index yang ada
    closest = similarity.columns[index[-1:-(k+2):-1]]
    
    # Drop movie_name agar nama movie yang dicari tidak muncul dalam daftar rekomendasi
    closest = closest.drop(movie_name, errors='ignore')
 
    return pd.DataFrame(closest).merge(items).head(k)

# %% [markdown]
# Mengecek fungsi movie_recommendations dan melihat hasilnya

# %%
movie_input = 'Toy Story (1995)'
print('Detail genre movie:\n\n', movie_dict[movie_dict.movie_name.eq(movie_input)])
print('\n\nResult top 5 movie recommendation:')
movie_recommendations(movie_input)

# %% [markdown]
# ### Collaborative Filtering

# %% [markdown]
# Membuat kelas RecommenderNet untuk Collaborative Filtering

# %%
class RecommenderNet(tf.keras.Model):
 
  # Insialisasi fungsi
  def __init__(self, num_users, num_movie, embedding_size, **kwargs):
    super(RecommenderNet, self).__init__(**kwargs)
    self.num_users = num_users
    self.num_movie = num_movie
    self.embedding_size = embedding_size
    self.user_embedding = layers.Embedding( # layer embedding user
        num_users,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.user_bias = layers.Embedding(num_users, 1) # layer embedding user bias
    self.movie_embedding = layers.Embedding( # layer embeddings movies
        num_movie,
        embedding_size,
        embeddings_initializer = 'he_normal',
        embeddings_regularizer = keras.regularizers.l2(1e-6)
    )
    self.movie_bias = layers.Embedding(num_movie, 1) # layer embedding movies bias
 
  def call(self, inputs):
    user_vector = self.user_embedding(inputs[:,0]) # memanggil layer embedding 1
    user_bias = self.user_bias(inputs[:, 0]) # memanggil layer embedding 2
    movie_vector = self.movie_embedding(inputs[:, 1]) # memanggil layer embedding 3
    movie_bias = self.movie_bias(inputs[:, 1]) # memanggil layer embedding 4
 
    dot_user_movie = tf.tensordot(user_vector, movie_vector, 2) 
 
    x = dot_user_movie + user_bias + movie_bias
    
    return tf.nn.sigmoid(x) # activation sigmoid

# %% [markdown]
# Mengcompile model

# %%
model = RecommenderNet(num_users, num_movie, 50) # inisialisasi model
 
# model compile
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)

# %% [markdown]
# Melakukan training model

# %%
# Memulai training
 
history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = 32,
    epochs = 25,
    validation_data = (x_val, y_val)
)

# %% [markdown]
# ## Evaluasi Model

# %% [markdown]
# Visualisasi metrik

# %%
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model_metrics')
plt.ylabel('root_mean_squared_error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %% [markdown]
# Mendapatkan Rekomendasi Movie

# %%
movie_df = movie_dict
df = pd.read_csv('datasets/ratings.csv')
 
# Mengambil sample user
user_id = df.userId.sample(1).iloc[0]
movie_visited_by_user = df[df.userId == user_id]
 
# Operator bitwise (~), bisa diketahui di sini https://docs.python.org/3/reference/expressions.html 
movie_not_visited = movie_df[~movie_df['id'].isin(movie_visited_by_user.movieId.values)]['id'] 
movie_not_visited = list(
    set(movie_not_visited)
    .intersection(set(movie_to_movie_encoded.keys()))
)
 
movie_not_visited = [[movie_to_movie_encoded.get(x)] for x in movie_not_visited]
user_encoder = user_to_user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movie_not_visited), movie_not_visited)
)

# %% [markdown]
# Rekomendasi Movie untuk User

# %%
ratings = model.predict(user_movie_array).flatten()
 
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [
    movie_encoded_to_movie.get(movie_not_visited[x][0]) for x in top_ratings_indices
]
 
print('Showing recommendations for users: {}'.format(user_id))
print('===' * 9)
print('Movie with high ratings from user')
print('----' * 8)
 
top_movie_user = (
    movie_visited_by_user.sort_values(
        by = 'rating',
        ascending=False
    )
    .head(5)
    .movieId.values
)
 
movie_df_rows = movie_df[movie_df['id'].isin(top_movie_user)]
for row in movie_df_rows.itertuples():
    print(row.movie_name, ':', row.genre)
 
print('----' * 8)
print('Top 10 movie recommendation')
print('----' * 8)
 
recommended_movie = movie_df[movie_df['id'].isin(recommended_movie_ids)]
for row in recommended_movie.itertuples():
    print(row.movie_name, ':', row.genre)



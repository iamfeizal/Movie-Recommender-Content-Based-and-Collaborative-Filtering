# Laporan Proyek Machine Learning - Imam Agus Faisal

## Domain Proyek
### Latar Belakang
Sistem rekomendasi film dengan *machine learning* adalah aplikasi yang membantu pengguna menemukan film yang sesuai dengan preferensi mereka. Sistem ini memanfaatkan berbagai algoritma *machine learning* dan data film yang tersedia untuk memberikan rekomendasi yang personal.<br>
Dalam sistem rekomendasi algoritma yang umum digunakan adalah *collaborative  filtering*(CF) dan *content based filtering*(CB). *collaborative filtering* adalah suatu konsep dimana opini dari pengguna lain yang ada digunakan untuk memprediksi item yang mungkin disukai/diminati oleh seorang pengguna. Sadangkan *content based filtering* menggunakan ketersediaan konten sebuah item sebagai basis dalam pemberian rekomendasi[^1].


### Mengapa Masalah Tersebut Harus Diselesaikan?
Penyelesaian masalah dalam pengembangan sistem rekomendasi film dengan machine learning memiliki beberapa alasan penting:
- **Meningkatkan Pengalaman Pengguna**: Masalah ini harus diselesaikan karena tujuan utama dari sistem rekomendasi adalah meningkatkan pengalaman pengguna. Dengan membantu pengguna menemukan konten yang sesuai dengan preferensi mereka, sistem ini dapat meningkatkan kepuasan pelanggan dan retensi pengguna.
- **Mengatasi Tantangan Eksplosi Konten**: Dalam era digital, jumlah konten film yang tersedia terus meningkat. Tanpa bantuan rekomendasi, pengguna mungkin merasa kewalahan dalam memilih apa yang harus mereka tonton. Sistem rekomendasi membantu mengatasi masalah ini dengan menyediakan pilihan yang relevan.

### Penelitian yang Dijadikan Referensi Yaitu:
- Dengan teknik penggabungkan secara *mixed hybrid* antara metode collaborative filtering dan *content-based filtering* dapat menghasilkan sistem rekomendasi yang mampu menutupi kekurangan dari setiap metode yang digunakan [^1].
- Algoritma *cosine similarity* digunakan untuk menghitung nilai kesamaan suatu produk. Metode *content-based filtering* dapat menyediakan rekomendasi *customer* [^2].
- Pendekatan *User Based Collaborative Filtering* sistem memberikan rekomendasi kepada *User* Item-Item yang disukai atau dinilai oleh User lain yang memiliki kemiripan. Kelebihan dari pendekantan *User Based Collaborative Filtering* adalah dapat menghasilkan rekomendasi yang berkualitas baik. Sedangkan kekurangannya adalah kompleksitas perhitungan akan semakin bertambah seiring dengan bertambahnya *User* sistem, semakin banyak *User* yang menggunakan system maka proses perekomendasian akan semakin lama [^3].

## Business Understanding
Dalam bagian ini dijelaskan mengenai manfaat dari pembuatan sistem rekomendasi film. Penjelasan mengenai aspek tersebut yaitu:
- **Personalisasi**: Pengguna memiliki preferensi yang beragam, dan solusi satu-ukuran-tidak-cocok untuk semua tidak memadai. Sistem rekomendasi memungkinkan personalisasi, yang dapat membantu menciptakan pengalaman yang lebih relevan dan memuaskan.
- **Kompetisi Bisnis**: Dalam industri hiburan, persaingan antara platform streaming film dan penyedia konten sangat sengit. Sistem rekomendasi yang baik dapat menjadi keunggulan kompetitif yang signifikan.

### Problem Statements
Berdasarkan latar belakang yang telah diuraikan sebelumnya, dikembangkanlah sebuah sistem rekomendasi film untuk menjawab permasalahan berikut:
- Bagaimana cara membuat rekomendasi film yang mempunyai karakteristik hampir sama dengan film yang pernah dilihat?
- Bagaimana cara merekomendasikan film berdasarkan preferensi pengguna?

### Goals
Untuk  menjawab pertanyaan tersebut, dibuatlah *system recommender* dengan tujuan atau *goals* sebagai berikut:
- Membuat sistem rekomendasi berdasarkan kesamaan film yang pernah dilihat yang akurat
- Membuat sistem rekomendasi yang dipersonalisasi berdasarkan preferensi pengguna

### Solution statements
- Menggunakan Metode *Content-Based Filtering* dengan *Cosine Similarity* untuk menyediakan rekomendasi film berdasarkan kesamaan karakteristik [^2].
- Menggunakan pendekatan *User Based Collaborative Filtering* sistem memberikan rekomendasi kepada *User* Item-Item yang disukai atau dinilai oleh User lain yang memiliki kemiripan [^3].

## Data Understanding
Menggunakan dataset dari Kaggle yaitu dataset *Movie Recommender System Dataset* dari [SHINIGAMI](https://www.kaggle.com/gargmanas).
Dataset ini memiliki 2 file csv yaitu: movies.csv dan ratings.csv

Sumber Dataset: [Movie Recommender System Dataset](https://www.kaggle.com/datasets/gargmanas/movierecommenderdataset/data).

### Variabel-variabel pada *Movie Recommender System Dataset* dataset adalah sebagai berikut:
- *movieId* : berisi unique id dari setiap film.
- *title* : berisi judul film tersebut beserta tahun dibuatnya.
- *genres* : berisi genre dari film tersebut.
- *userId* : berisi unique id dari setiap user yang memberi rating film.
- *rating* : berisi nilai rating user untuk sebuah film (0.5-5).
- *timestamp* : berisi waktu saat user memberi rating (dalam detik).


### Exploratory Data Analysis:
Dalam bagian ini akan dijelaksan mengenai analisis univariate dan multivariate, yaitu:
- **Univariate analysis** bertujuan untuk memahami dan menganalisis satu variabel atau fitur pada suatu waktu. Ini digunakan untuk menggambarkan distribusi, karakteristik, dan statistik dasar dari satu variabel tertentu.
- **Multivariate analysis** bertujuan untuk memahami hubungan kompleks antara dua atau lebih variabel dalam dataset. Ini digunakan untuk mengeksplorasi bagaimana variabel-variabel tersebut berinteraksi satu sama lain.

#### Univariate Analysis
| ![](/assets/images/film_pertahun.png) <center><b>Gambar 1</b> - Jumlah Film yang Rilis Setiap Tahun</center> | ![](/assets/images/rating_pertahun.png) <center><b>Gambar 2</b> - Jumlah Rating yang Diberikan Setiap Tahun</center>  |
|---|---|

**Kesimpulan:**<br>
Perilisan film paling banyak terjadi pada tahun 2002, sedangkan jumlah pemberian rating paling banyak terjadi pada tahun 2000.

#### Multivariate Analysis
| ![](/assets/images/film_rating_terbanyak.png) <center><b>Gambar 3</b> - 10 Film dengan Jumlah Rating Paling Banyak</center> | ![](/assets/images/distribusi_rating.png) <center><b>Gambar 4</b> - Distribusi Rating pada Semua Film</center>  |
|---|---|

**Kesimpulan:**<br>
Film Forest Gump (1994) memilki jumlah rating paling banyak, sedangkan distribusi rating paling banyak terdapat pada nilai rating 4. 

## Data Preparation
Pada tahap Data *Preparation* dibagi menjadi 2 tahapan, yaitu:
- Data *Preparation* untuk *Content Based Filtering*
- Data *Preparation* untuk *Collaborative Filtering*

### Data Preparation untuk Content Based Filtering
- Mengatasi *Missing Value*<br>Mengecek apakah terdapat *missing value* pada data atau tidak.
- Memisahkan Tahun Pembuatan Film<br>Agar lebih mudah dalam menganalisis data dan mengelompokkannya berdasarkan tahun rilis.
- Mengatasi *Duplicated* Data<br>Mengecek apakah terdapat data yang sama persis didalam *dataframe*.
- Mengkonversi Data Series Menjadi List<br>Agar data dapat dipasangkan ke dalam *dictionary*
- Membuat *Dictionary*<br>Untuk menentukan pasangan Key-Value.

### Data Preparation untuk Collaborative Filtering
- Melakukan *Encoding*<br>Agar data dapat dimasukkan ke dalam *integer*.
- Memetakan data kedalam Dataframe
- Membagi data ke dalam data training dan validasi dengan rasio 80:20


## Modeling
Menggunakan 2 metode algoritma machine learning untuk sistem rekomendasi film, yaitu:
- **Content-Based Filtering**<br>Metode *Content-Based Filtering* dengan *Cosine Similarity* digunakan untuk menyediakan rekomendasi film berdasarkan kesamaan karakteristik [^2].<br>
**Tahap TF-IDF Vectorizer**<br>
Pada dataframe film, dilakukan pembobotan dengan TF-IDF Vectorizer untuk menemukan representasi fitur penting dari setiap genre film.<br>
Untuk menghasilkan vektor tf-idf dalam bentuk matriks, gunakan fungsi todense() dalam library scipy.<br>
Semakin tinggi nilai pada matriks, maka semakin cocok judul film dengan genre tersebut.<br>
**Tahap Cosine Similarity**<br>
Menghitung cosine similarity dataframe tfidf_matrix yang diperoleh pada tahapan sebelumnya menggunakan fungsi cosine_similarity dari library sklearn.

  |                            movie_name | Mirror Mirror (2012) | Message in a Bottle (1999) | Howards End (1992) | Masquerade (1988) | Meatballs Part II (1984) |
  |--------------------------------------:|---------------------:|---------------------------:|-------------------:|------------------:|-------------------------:|
  | Wizards of the Lost Kingdom II (1989) |             0.556470 |                   0.000000 |           0.000000 |          0.000000 |                 0.000000 |
  |                           DiG! (2004) |             0.000000 |                   0.000000 |           0.000000 |          0.000000 |                 0.000000 |
  |                        Airport (1970) |             0.000000 |                   0.000000 |           1.000000 |          0.000000 |                 0.000000 |
  |                   Mystic Pizza (1988) |             0.195131 |                   0.726418 |           0.466539 |          0.375645 |                 0.504636 |
  | Dragonheart 2: A New Beginning (2000) |             0.771308 |                   0.000000 |           0.275732 |          0.195724 |                 0.298247 |
  
  **Semakin tinggi nilai *cosine similarity*, maka semakin mirip karakteristik film tersebut.**<br>

  **Membuat Fungsi movie_recommendations**
  membuat fungsi movie_recommendations dengan beberapa parameter sebagai berikut:
  - movie_name : Nama judul dari movie tersebut (index kemiripan dataframe).
  - similarity_data : Dataframe mengenai similarity yang telah kita didefinisikan sebelumnya
  - items : Nama dan fitur yang digunakan untuk mendefinisikan kemiripan, dalam hal ini adalah ‘movie_name’ dan ‘genre’.
  - k : Banyak rekomendasi yang ingin diberikan.<br>
<br>

  **Hasil Content-Based Filtering:**

  Detail genre movie:
  |   | id | movie_name       | genre                                           |
  |---|----|------------------|-------------------------------------------------|
  | 0 | 1  | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |

  Top 5 movie recommendation:
  |   |                                     movie_name |                                           genre |
  |---|-----------------------------------------------:|------------------------------------------------:|
  | 0 |               Emperor's New Groove, The (2000) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
  | 1 |                             Toy Story 2 (1999) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
  | 2 |                                   Turbo (2013) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
  | 3 |                         Shrek the Third (2007) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
  | 4 | Adventures of Rocky and Bullwinkle, The (2000) | Adventure\|Animation\|Children\|Comedy\|Fantasy |

  **Kesimpulan:**<br>
  Berdasarkan hasil diatas, dapat dilihat bahwa terdapat top 5 film yang memiliki kesamaan dengan Toy Story (1995). Hal tersebut didasarkan pada nilai *Cosine Similarity* pada film Toy Story (1995) dengan film lainnya dan didapatkanlah Top 5 film yang memiliki tingkat kesaman paling tinggi.<br><br>

- **Collaborative Filtering**<br>Kelebihan dari pendekantan *User Based Collaborative Filtering* adalah dapat menghasilkan rekomendasi yang berkualitas baik. Sedangkan kekurangannya adalah kompleksitas perhitungan akan semakin bertambah seiring dengan bertambahnya *User* sistem, semakin banyak *User* yang menggunakan system maka proses perekomendasian akan semakin lama [^3].<br>
  **Membuat Kelas RecommenderNet**<br>
  membuat kelas RecommenderNet dengan beberapa parameter sebagai berikut:
  - tf.keras.Model : mengambil Model dari library TensorFlow Keras.
  - num_users : jumlah user dari data user yang sudah di encoding
  - num_movie : jumlah movie dari data user yang sudah di encoding
  - embedding_size : ukuran layer embedding.<br>
  **Compile Model**<br>
  Dalam mengcompline model ini menggunakan Binary Crossentropy untuk menghitung loss function, Adam (Adaptive Moment Estimation) sebagai optimizer dengan learning_rate sebesar 0.001, dan root mean squared error (RMSE) sebagai metrics evaluation. 
  
  **Hasil Collaborative Filtering:**

  | Showing recommendations for users: 307                                |
  |-----------------------------------------------------------------------|
  | ===========================<br>Movie with high ratings from user<br>--------------------------------<br>American History X (1998) : Crime\|Drama<br>Eternal Sunshine of the Spotless Mind (2004) : Drama\|Romance\|Sci-Fi<br>Shaun of the Dead (2004) : Comedy\|Horror<br>Reign Over Me (2007) : Drama<br>Hangover, The (2009) : Comedy\|Crime<br>--------------------------------<br>Top 10 movie recommendation<br>--------------------------------<br>Underground (1995) : Comedy\|Drama\|War<br>Streetcar Named Desire, A (1951) : Drama<br>Cinema Paradiso (Nuovo cinema Paradiso) (1989) : Drama<br>Paths of Glory (1957) : Drama\|War<br>Touch of Evil (1958) : Crime\|Film-Noir\|Thriller<br>Night on Earth (1991) : Comedy\|Drama<br>Double Indemnity (1944) : Crime\|Drama\|Film-Noir<br>Trial, The (Procès, Le) (1962) : Drama<br>Day of the Doctor, The (2013) : Adventure\|Drama\|Sci-Fi<br>Three Billboards Outside Ebbing, Missouri (2017) : Crime\|Drama |

  **Kesimpulan:**<br>
  Berdasarkan hasil diatas yang menampilkan rekomendasi untuk user 307. Hasil tersebut diperoleh dari hasil personalisasi user 307 saat memberikan rating tinggi pada film.


## Evaluation
### Content-Based Filtering
Rumus evaluasi menggunakan precission :
$$Precision = \frac{our recommendarion that are relevan}{items we recommended}$$

  Detail genre movie:
  |   | id | movie_name       | genre                                           |
  |---|----|------------------|-------------------------------------------------|
  | 0 | 1  | Toy Story (1995) | Adventure\|Animation\|Children\|Comedy\|Fantasy |

  Top 5 movie recommendation:
  |   |                                     movie_name |                                           genre |
  |---|-----------------------------------------------:|------------------------------------------------:|
  | 0 |               Emperor's New Groove, The (2000) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
  | 1 |                             Toy Story 2 (1999) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
  | 2 |                                   Turbo (2013) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
  | 3 |                         Shrek the Third (2007) | Adventure\|Animation\|Children\|Comedy\|Fantasy |
  | 4 | Adventures of Rocky and Bullwinkle, The (2000) | Adventure\|Animation\|Children\|Comedy\|Fantasy |

  **Kesimpulan:**<br>
  Dari hasil rekomendasi di atas, diketahui bahwa Toy Story (1995) termasuk ke dalam genre Adventure\|Animation\|Children\|Comedy\|Fantasy. Dari 5 item yang direkomendasikan, semua item memiliki genre Adventure\|Animation\|Children\|Comedy\|Fantasy (similar).<br>**Artinya, precision sistem pada film Toy Story (1995) sebesar 5/5 atau 100%.**

### Collaborative Filtering
Evaluasi metrik yang digunakan untuk mengukur kinerja model adalah metrik RMSE (Root Mean Squared Error). RMSE adalah akar kuadrat dari rata-rata dari kuadrat selisih antara prediksi model (y_pred) dan nilai aktual (y_true). Rumus RMSE adalah sebagai berikut:
$$RMSE = \sqrt{\frac{\sum_{t=1}^{n}(y\_pred_t-y\_true_t)^2}{n}}$$
Di sini, y_true adalah nilai aktual, y_pred adalah nilai yang diprediksi oleh model, dan n adalah jumlah sampel (data points) dalam dataset.

**Keuntungan RMSE:**
- Sensitif terhadap Kesalahan Besar: RMSE akan memberikan bobot yang lebih besar pada kesalahan yang signifikan. Ini penting karena kesalahan besar biasanya lebih kritis dalam aplikasi dunia nyata.

**Kerugian RMSE:**
- Sensitif terhadap Outliers: RMSE sensitif terhadap data outlier. Outliers yang ekstrim dapat memiliki pengaruh yang signifikan pada hasil RMSE.
- Satuan yang Sama: RMSE menyajikan kesalahan dalam satuan yang sama dengan variabel target, yang bisa sulit untuk diinterpretasikan dalam beberapa kasus.

Hasil:<br>
![](/assets/images/rmse.png) <center><b>Gambar 5</b> - Evaluasi RMSE</center>

**Kesimpulan:**<br>
Berdasarkan gambar 5, nilai RMSE pada model dengan 25 epochs kian menurun. Hal tersebut menandakan bahwa model memiliki akurasi yang semakin baik. Selain itu, grafik yang pada awalnya mengalami penurunan kemudian dilanjutkan dengan menunjukkan kestabilan menandakan model sudah menunjukkan hasil goodfit.

## Daftar Referensi

[^1]: [A. E. Wijaya, D. Alfian, "SISTEM REKOMENDASI LAPTOP MENGGUNAKAN COLLABORATIVE FILTERING DAN CONTENT-BASED FILTERING," Jurnal Computech & Bisnis, vol.12, no.1, pp.11-27, 2018.](https://scholar.archive.org/work/hlmop7z6g5de3bzspxkz6fbl7u/access/wayback/http://www.jurnal.stmik-mi.ac.id:80/index.php/jcb/article/download/167/189)

[^2]: [F. B. A. Larasati, H. Februariyanti, "SISTEM REKOMENDASI PRODUCT EMINA COSMETICS DENGAN MENGGUNAKAN METODE CONTENT - BASED FILTERING," MISI (Jurnal Manajemen informatika & Sistem Informasi), vol.4, no.1, pp.45-54, 2021](https://www.e-journal.stmiklombok.ac.id/index.php/misi/article/download/250/131)

[^3]: [A. N. Khusna, K. P. Delasano, D. C. E. Saputra, "Penerapan User-Based Collaborative Filtering Algorithm Studi Kasus Sistem Rekomendasi untuk Menentukan Gadget Shield," Matrik: Jurnal Manajemen, Teknik Informatika, dan Rekayasa Komputer, vol.20, no.2, pp.293-304, 2021](https://journal.universitasbumigora.ac.id/index.php/matrik/article/download/1124/691)

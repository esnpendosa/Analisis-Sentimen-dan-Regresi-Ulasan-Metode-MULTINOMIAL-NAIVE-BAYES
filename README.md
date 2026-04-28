# Analisis Sentimen Ulasan Aplikasi (Multinomial Naive Bayes)

Proyek ini adalah sistem analisis sentimen untuk ulasan aplikasi **Wondr by BNI** yang diambil dari Play Store. Sistem ini telah distandarisasi menggunakan algoritma **Multinomial Naive Bayes** dengan pipeline pemrosesan teks bahasa Indonesia yang lengkap.

---

## 🚀 Fitur Utama
- **Preprocessing 6 Langkah**: Cleaning, Case Folding, Normalisasi (Slang), Tokenizing, Stopword Removal, dan Stemming (Sastrawi).
- **Pembobotan TF-IDF**: Ekstraksi fitur teks yang akurat.
- **Klasifikasi Multinomial Naive Bayes**: Algoritma yang efisien untuk klasifikasi teks.
- **Visualisasi Real-time**: Menampilkan progres tiap langkah preprocessing di aplikasi web.
- **Tab Evaluasi Sistem**: Menampilkan Confusion Matrix dan metrik performa (Akurasi, Presisi, Recall, F1-Score).

---

## 🛠️ Langkah-Langkah Sistem (Standar)

### Langkah Ke-satu: Preprocessing
1. **Cleaning**: Membersihkan mention, hashtag, URL, angka, dan tanda baca.
2. **Case Folding**: Mengubah teks menjadi huruf kecil.
3. **Normalisasi**: Memperbaiki kata tidak baku (slang) menjadi kata formal.
4. **Tokenizing**: Pemecahan kalimat menjadi kata-kata tunggal.
5. **Stopword Removal**: Menghapus kata-kata umum yang tidak bermakna.
6. **Stemming**: Mengubah kata berimbuhan menjadi kata dasar menggunakan Sastrawi.

### Langkah Ke-dua: Pembobotan Kata
Menggunakan **TF-IDF Vectorizer** untuk mengubah data teks yang telah bersih menjadi representasi angka.

### Langkah Ke-tiga: Klasifikasi
Menggunakan algoritma **Multinomial Naive Bayes** untuk memprediksi kategori sentimen (Positif, Netral, atau Negatif).

### Langkah Ke-empat: Evaluasi
Mengukur performa sistem menggunakan **Confusion Matrix** dan menghitung skor Akurasi, Presisi, Recall, serta F1-Score.

---

## 💻 Cara Menjalankan

### 1. Persiapan (Setup)
Instal semua library yang dibutuhkan:
```powershell
pip install -r requirements.txt
```

### 2. Pelatihan Model (Training)
Anda bisa melatih model secara otomatis untuk memperbarui file `.h5` dan `.png`:
```powershell
& "C:\Users\MSI\AppData\Local\Programs\Python\Python312\python.exe" train_auto.py
```
*Atau jalankan seluruh sel di file `[Sentiment].ipynb`.*

### 3. Menjalankan Aplikasi Web
Jalankan antarmuka Streamlit:
```powershell
& "C:\Users\MSI\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run app.py
```

---

## 📂 Struktur Proyek
- `app.py`: Aplikasi web utama (Streamlit).
- `[Sentiment].ipynb`: Notebook riset dan evaluasi mendalam.
- `train_auto.py`: Script otomatisasi pelatihan model.
- `confusion_matrix.png`: Hasil visualisasi evaluasi sistem.
- `model_nb_tfidf.h5`: Model Naive Bayes yang sudah dilatih.
- `tfidf_vectorizer.pkl`: Vectorizer TF-IDF yang sudah dilatih.
- `reviews_wondr_bni.csv`: Dataset ulasan aplikasi.

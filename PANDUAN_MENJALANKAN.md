# Panduan Menjalankan Proyek Analisis Sentimen (Naive Bayes)

Panduan ini menjelaskan cara menjalankan sistem Analisis Sentimen ulasan aplikasi **Wondr by BNI** menggunakan algoritma **Multinomial Naive Bayes**.

---

## 1. Persiapan Lingkungan (Setup)
Sebelum menjalankan program, pastikan Anda sudah menginstal semua pustaka (library) yang diperlukan. Buka terminal/CMD di folder proyek dan jalankan:

```powershell
pip install -r requirements.txt
```

---

## 2. Tahap Pelatihan Model (Training)
Anda harus melatih model untuk menghasilkan file model (`.h5`) dan gambar evaluasi (`.png`).

1. **Otomatis**: Jalankan script pelatihan otomatis:
   ```powershell
   & "C:\Users\MSI\AppData\Local\Programs\Python\Python312\python.exe" train_auto.py
   ```
2. **Manual**: Buka file `[Sentiment].ipynb` menggunakan Jupyter Notebook/VS Code dan klik **"Run All"**.

*Setelah selesai, file `model_nb_tfidf.h5`, `tfidf_vectorizer.pkl`, dan `confusion_matrix.png` akan diperbarui.*

---

## 3. Menjalankan Aplikasi Web (Streamlit)
Setelah model siap, jalankan antarmuka web dengan perintah berikut:

```powershell
& "C:\Users\MSI\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run app.py
```

---

## 4. Cara Penggunaan Aplikasi
1. Buka link yang muncul di terminal (biasanya `http://localhost:8501`).
2. Masukkan teks ulasan pada kotak yang disediakan.
3. Klik tombol **"Analisis Sentimen Sekarang"**.
4. Sistem akan menampilkan **visualisasi 6 langkah preprocessing** (Cleaning sampai Stemming).
5. Hasil prediksi (Positif/Netral/Negatif) akan muncul di bawahnya.
6. Pindah ke tab **"Performa Model (Langkah 4)"** untuk melihat **Confusion Matrix** dan metrik evaluasi sistem.

---

## 5. Struktur Penting
- `[Sentiment].ipynb`: Notebook riset dan training.
- `app.py`: Aplikasi web utama (Streamlit).
- `train_auto.py`: Script otomatisasi training.
- `confusion_matrix.png`: Hasil visualisasi evaluasi sistem.
- `requirements.txt`: Daftar library yang dibutuhkan.
- `reviews_wondr_bni.csv`: Dataset ulasan.

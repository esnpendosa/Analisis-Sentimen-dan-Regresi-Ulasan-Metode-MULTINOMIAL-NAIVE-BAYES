import streamlit as st
import pandas as pd
import joblib
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import numpy as np

# --- KONFIGURASI HALAMAN (Harus menjadi perintah st pertama) ---
st.set_page_config(page_title="Analisis Sentimen", page_icon="📊", layout="wide")


# --- FUNGSI PREPROCESSING & LOAD MODEL (Menggunakan cache agar lebih cepat) ---

@st.cache_data
def load_stopwords():
    stopword_df = pd.read_csv('stopwordbahasa.csv', header=None, names=['stopwords'])
    return set(stopword_df['stopwords'].values)

@st.cache_resource
def get_stemmer():
    factory = StemmerFactory()
    return factory.create_stemmer()

@st.cache_resource
def load_models():
    """
    MEMUAT MODEL MULTINOMIAL NAIVE BAYES
    Metode ini memuat file model yang sudah dilatih di notebook [Sentiment].ipynb
    """
    try:
        # Memuat model Multinomial Naive Bayes (.h5 atau .pkl)
        model_klasifikasi = joblib.load('model_nb_tfidf.h5')
        
        # Model regresi dinonaktifkan (fokus pada Naive Bayes sesuai request)
        model_regresi = None
        
        # Memuat Vectorizer TF-IDF untuk mengubah teks menjadi angka
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        
        return model_klasifikasi, model_regresi, tfidf_vectorizer
    except FileNotFoundError:
        st.error("File model 'model_nb_tfidf.h5' tidak ditemukan!")
        return None, None, None

# Muat komponen utama
stopwords_id = load_stopwords()
stemmer = get_stemmer()
model_klasifikasi, model_regresi, tfidf_vectorizer = load_models()

# Kamus Normalisasi (Slang to Formal Indonesian)
norm_dict = {
    "yg": "yang", "gk": "tidak", "tdk": "tidak", "bgt": "banget", "gpp": "tidak apa-apa",
    "kl": "kalau", "klo": "kalau", "udah": "sudah", "sdh": "sudah", "aja": "saja",
    "ga": "tidak", "gak": "tidak", "kmrn": "kemarin", "skrg": "sekarang", "tp": "tapi",
    "dgn": "dengan", "dlm": "dalam", "utk": "untuk", "bisaa": "bisa", "mantap": "bagus",
    "oke": "baik", "ok": "baik", "sip": "baik", "kalo": "kalau", "biar": "supaya",
    "krn": "karena", "karna": "karena", "bngt": "banget", "udh": "sudah", "sy": "saya"
}

# --- FUNGSI PREPROCESSING (6 LANGKAH STANDAR INDONESIA) ---
def preprocess_text(text):
    # LANGKAH 1: Cleaning (Membersihkan teks dari URL, mention, hashtag, angka, dan tanda baca)
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    
    # LANGKAH 2: Case Folding (Mengubah teks menjadi huruf kecil semua)
    text = text.lower()
    
    # LANGKAH 3: Normalisasi (Mengubah kata tidak baku/slang menjadi kata formal)
    words = text.split()
    words = [norm_dict.get(word, word) for word in words]
    
    # LANGKAH 4: Tokenizing (Memecah teks menjadi daftar kata/token - dilakukan saat split di atas)
    
    # LANGKAH 5: Stopword Removal (Menghapus kata-kata umum yang tidak memiliki makna penting)
    words = [word for word in words if word not in stopwords_id]
    
    # LANGKAH 6: Stemming (Mengembalikan kata ke bentuk dasarnya menggunakan library Sastrawi)
    words = [stemmer.stem(word) for word in words]
    
    return " ".join(words)


# --- TAMPILAN UTAMA APLIKASI WEB (UI) ---

st.title("📊 Analisis Sentimen Ulasan Aplikasi")
st.markdown("Aplikasi ini menggunakan metode **Multinomial Naive Bayes** dengan alur kerja standar NLP.")

# Membuat Tab untuk Navigasi
tab1, tab2 = st.tabs(["🔍 Analisis Sentimen", "📈 Performa Model (Langkah 4)"])

with tab1:
    st.subheader("Input Ulasan")
    input_review = st.text_area("Masukkan teks ulasan di sini:", height=150, placeholder="Contoh: Aplikasinya bagus dan sangat membantu!")

    if st.button("Analisis Sentimen Sekarang"):
        if all([model_klasifikasi, tfidf_vectorizer]) and input_review:
            
            # --- LANGKAH KE-SATU: PREPROCESSING (6 TAHAP) ---
            st.info("🔄 Menjalankan Langkah Ke-satu: Preprocessing...")
            
            status_placeholder = st.empty()
            
            # Tahap 1: Cleaning
            status_placeholder.text("1. Menjalankan Cleaning...")
            text_cleaned = re.sub(r'@[A-Za-z0-9_]+', '', input_review)
            text_cleaned = re.sub(r'#[A-Za-z0-9_]+', '', text_cleaned)
            text_cleaned = re.sub(r'http\S+|www.\S+', '', text_cleaned)
            text_cleaned = re.sub(r'\d+', '', text_cleaned)
            text_cleaned = text_cleaned.translate(str.maketrans('', '', string.punctuation))
            text_cleaned = re.sub(r'\s+', ' ', text_cleaned).strip()
            
            # Tahap 2: Case Folding
            status_placeholder.text("2. Menjalankan Case Folding...")
            text_folded = text_cleaned.lower()
            
            # Tahap 3: Normalisasi
            status_placeholder.text("3. Menjalankan Normalisasi Kata...")
            words = text_folded.split()
            words_norm = [norm_dict.get(word, word) for word in words]
            
            # Tahap 4: Tokenizing
            status_placeholder.text("4. Menjalankan Tokenizing...")
            # (Sudah dipecah menjadi words_norm)
            
            # Tahap 5: Stopword Removal
            status_placeholder.text("5. Menjalankan Stopword Removal...")
            words_nostop = [word for word in words_norm if word not in stopwords_id]
            
            # Tahap 6: Stemming
            status_placeholder.text("6. Menjalankan Stemming (Sastrawi)...")
            words_stemmed = [stemmer.stem(word) for word in words_nostop]
            text_final = " ".join(words_stemmed)
            
            status_placeholder.success("✅ Preprocessing Selesai!")

            # --- LANGKAH KE-DUA: PEMBOBOTAN TF-IDF ---
            st.info("🔢 Menjalankan Langkah Ke-dua: Pembobotan Kata (TF-IDF)...")
            X_new = tfidf_vectorizer.transform([text_final])

            # --- LANGKAH KE-TIGA: MULTINOMIAL NAIVE BAYES ---
            st.info("🧠 Menjalankan Langkah Ke-tiga: Perhitungan Multinomial Naive Bayes...")
            pred_klasifikasi = model_klasifikasi.predict(X_new)[0]
            
            # Tampilkan Hasil Utama
            st.divider()
            st.subheader("Hasil Analisis:")
            if pred_klasifikasi == 'positif':
                st.success(f"Sentimen Terdeteksi: **{pred_klasifikasi.upper()}** 👍")
            elif pred_klasifikasi == 'negatif':
                st.error(f"Sentimen Terdeteksi: **{pred_klasifikasi.upper()}** 👎")
            else:
                st.warning(f"Sentimen Terdeteksi: **{pred_klasifikasi.upper()}** 😐")
            
            # Menampilkan Teks Hasil Olahan
            with st.expander("Lihat Detail Hasil Preprocessing"):
                st.write(f"**Teks Awal:** {input_review}")
                st.write(f"**Teks Akhir (Hasil 6 Langkah):** {text_final}")

        elif not input_review:
            st.warning("Mohon masukkan teks ulasan terlebih dahulu.")

with tab2:
    st.header("Evaluasi Sistem (Langkah Ke-empat)")
    st.write("Berikut adalah performa model Multinomial Naive Bayes berdasarkan hasil pengujian:")
    
    # Data metrik (berdasarkan hasil training notebook)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi", "85.80%")
    col2.metric("Presisi", "0.5523")
    col3.metric("Recall", "0.5741")
    col4.metric("F1-Score", "0.5628")
    
    st.divider()
    st.subheader("Confusion Matrix")
    
    import os
    if os.path.exists("confusion_matrix.png"):
        st.image("confusion_matrix.png", caption="Confusion Matrix - Naive Bayes", width='stretch')
    else:
        st.warning("⚠️ File 'confusion_matrix.png' tidak ditemukan. Silakan jalankan notebook [Sentiment].ipynb untuk menghasilkan gambar ini.")
    
    st.info("Catatan: Metrik di atas dihitung menggunakan data uji (test set) sebesar 25% dari dataset.")
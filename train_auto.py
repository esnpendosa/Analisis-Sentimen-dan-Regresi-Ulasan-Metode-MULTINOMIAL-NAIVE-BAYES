import pandas as pd
import re
import string
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

print("Membaca dataset...")
df = pd.read_csv("reviews_wondr_bni.csv")
df = df[['content', 'score']].dropna()

# Ambil sample 10.000 (sesuai notebook)
df_sampled = df.sample(n=10000, random_state=42).reset_index(drop=True)

# Preprocessing Setup
stopword_df = pd.read_csv('stopwordbahasa.csv', header=None, names=['stopwords'])
stopwords_id = set(stopword_df['stopwords'].values)
factory = StemmerFactory()
stemmer = factory.create_stemmer()

norm_dict = {
    "yg": "yang", "gk": "tidak", "tdk": "tidak", "bgt": "banget", "gpp": "tidak apa-apa",
    "kl": "kalau", "klo": "kalau", "udah": "sudah", "sdh": "sudah", "aja": "saja",
    "ga": "tidak", "gak": "tidak", "kmrn": "kemarin", "skrg": "sekarang", "tp": "tapi",
    "dgn": "dengan", "dlm": "dalam", "utk": "untuk", "bisaa": "bisa", "mantap": "bagus",
    "sy": "saya", "udh": "sudah"
}

def preprocess(text):
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    words = [norm_dict.get(w, w) for w in text.split()]
    words = [w for w in words if w not in stopwords_id]
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

print("Memulai Preprocessing 6 Langkah... (Sastrawi mungkin agak lambat)")
df_sampled['text_final'] = df_sampled['content'].astype(str).apply(preprocess)
df_sampled['sentiment'] = df_sampled['score'].apply(lambda x: 'positif' if x >= 4 else ('netral' if x == 3 else 'negatif'))

print("Ekstraksi Fitur TF-IDF...")
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df_sampled['text_final'])
y = df_sampled['sentiment']

print("Pelatihan Model Naive Bayes...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

print("Menghasilkan Confusion Matrix...")
y_pred = nb_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=nb_model.classes_, yticklabels=nb_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Naive Bayes')
plt.savefig('confusion_matrix.png')

print("Menyimpan Model...")
joblib.dump(nb_model, 'model_nb_tfidf.h5')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("Selesai! Model dan Gambar berhasil dibuat.")

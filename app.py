import re

import joblib
import streamlit as st

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
except ModuleNotFoundError:
    st.error("Package Sastrawi belum terpasang. Jalankan: pip install Sastrawi")
    st.stop()

model = joblib.load("model_naive_bayes_random_oversampler.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
stemmer = StemmerFactory().create_stemmer()

custom_stopwords = {
    "yang", "dan", "di", "ke", "dari", "untuk", "dengan", "ini", "itu", "karena",
    "ada", "juga", "saja", "nya", "aja", "atau", "pada", "jadi", "sudah", "udah",
    "kalau", "kalo", "sangat", "banget", "bgt", "buat", "sama", "dalam", "sebagai",
    "agar", "lebih", "kurang", "bisa", "tidak", "ga", "gak", "nggak", "ngga", "tp",
    "tapi", "karna", "krn", "yg", "dr", "nih", "deh", "sih", "ya", "iya", "lah",
    "pun", "the", "of", "is", "in", "to", "tempat", "wisata"
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = [token for token in text.split() if len(token) > 1]
    tokens = [token for token in tokens if token not in custom_stopwords]
    text = " ".join(tokens)
    text = stemmer.stem(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


st.set_page_config(page_title="Analisis Sentimen D'LAS", page_icon="📊", layout="centered")
st.title("Analisis Sentimen Ulasan Wisata D'LAS")
st.caption("Model: TF-IDF + Multinomial Naive Bayes + RandomOverSampler")

user_input = st.text_area(
    "Masukkan ulasan",
    height=160,
    placeholder="Contoh: tempatnya bagus tapi jalan menuju lokasi rusak"
)

if st.button("Prediksi"):
    if not user_input.strip():
        st.warning("Masukkan ulasan terlebih dahulu.")
    else:
        cleaned_text = clean_text(user_input)
        vector = vectorizer.transform([cleaned_text])
        prediction = model.predict(vector)[0]

        st.subheader("Hasil Sentimen")
        st.write(prediction)

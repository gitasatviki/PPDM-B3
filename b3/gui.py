import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import gensim
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
import csv

nltk.download('punkt')
nltk.download('stopwords')

with open('model_svm_word2vec.pkl', 'rb') as f:
    svm_word2vec_best = pickle.load(f)

with open('model_word2vec.pkl', 'rb') as f:
    model_w2v = pickle.load(f)

def plot_results(positive_word2vec, negative_word2vec):
    labels = ['Word2Vec Positif', 'Word2Vec Negatif']
    values = [positive_word2vec, negative_word2vec]

    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['green', 'red'])
    ax.set_ylabel('Count')
    ax.set_title('Sentiment Analysis Results')
    st.pyplot(fig)

def cleaning_ulasan(ulasan):
    ulasan = re.sub(r'@[A-Za-z0-9]+', ' ', ulasan)
    ulasan = re.sub(r'#[A-Za-z0-9]+', ' ', ulasan)
    ulasan = re.sub(r"http\S+", ' ', ulasan)
    ulasan = re.sub(r'[0-9]+', ' ', ulasan)
    ulasan = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", ulasan)
    ulasan = ulasan.strip(' ')
    return ulasan

def clear_emoji(ulasan):
    return ulasan.encode('ascii', 'ignore').decode('ascii')

def double(ulasan):
    pattern = re.compile(r'(.)\1{2,}', re.DOTALL)
    return pattern.sub(r'\1', ulasan)

def case_folding_text(ulasan):
    return ulasan.lower()

def tokenizing_text(ulasan):
    return word_tokenize(ulasan)

def convert_to_slangword(ulasan):
    with open("slangwords.txt", "r") as file:
        kamus_slang = eval(file.read())
    pattern = re.compile(r'\b(' + '|'.join(kamus_slang.keys()) + r')\b')
    content = []
    for kata in ulasan:
        filter_slang = pattern.sub(lambda x: kamus_slang[x.group()], kata)
        content.append(filter_slang.lower())
    ulasan = content
    return ulasan

daftar_stopword = stopwords.words('indonesian')
daftar_stopword = set(daftar_stopword)
def stopword_text(words):
    return [word for word in words if word not in daftar_stopword]

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemmed_wrapper(term):
    return stemmer.stem(term)

def preprocessing_pipeline(ulasan):
    ulasan = cleaning_ulasan(ulasan)
    ulasan = clear_emoji(ulasan)
    ulasan = double(ulasan)
    ulasan = case_folding_text(ulasan)
    ulasan = tokenizing_text(ulasan)
    ulasan = convert_to_slangword(ulasan)
    ulasan = stopword_text(ulasan)
    ulasan = [stemmed_wrapper(term) for term in ulasan]
    return ulasan

def word_vector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += model_w2v.wv[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

st.title('Aplikasi Analisis Ulasan Sentimen Google Play Store')
st.subheader("Kelompok B3")
st.caption( "1. Ni Made Gita Satviki Nirmala (2208561053)\n"
            "2. I Wayan Restama Yasa  (2208561027)\n"
            "3. Danendra Darmawansyah (2208561091)\n"
            "4. Ida Bagus Gde Ardita Mahaprawira  (2208561127)\n")
st.divider()

uploaded_file = st.file_uploader("Unggah File", type=["csv"])
text_input = st.text_area("Atau Masukkan Kalimat")

col1, col2 = st.columns(2)

if uploaded_file is not None:
    ulasan = pd.read_csv(uploaded_file)
    st.subheader("Data Awal Sebelum Dilakukan Analisis:")
    st.write(ulasan) 
    ulasan.dropna(inplace=True)
    if not ulasan.empty:
        ulasan['processed'] = ulasan['review'].apply(preprocessing_pipeline)
        st.subheader("Data Setelah Preprocessing:")
        st.write(ulasan)
        ulasan_text = [' '.join(tokens) for tokens in ulasan['processed']]
        x_word2vec = np.concatenate([word_vector(tokens, 400) for tokens in ulasan['processed']])
        
        word2vec_predictions = svm_word2vec_best.predict(x_word2vec)

        positive_word2vec = sum(word2vec_predictions == 'Positif')
        negative_word2vec = sum(word2vec_predictions == 'Negatif')

        st.write("Word2Vec dengan Support Vector Machine:")
        st.write(f"Positif: {positive_word2vec} ({positive_word2vec / len(word2vec_predictions) * 100:.2f}%)")
        st.write(f"Negatif: {negative_word2vec} ({negative_word2vec / len(word2vec_predictions) * 100:.2f}%)")

        plot_results(positive_word2vec, negative_word2vec)

if col1.button("Analisis"):
    if uploaded_file is not None:
        pass

if col2.button("Reset"):
    uploaded_file = None
    text_input = None

if text_input:
    processed_text = preprocessing_pipeline(text_input)
    if processed_text:
        st.subheader("Hasil Preprocessing untuk Kalimat yang Dimasukkan:")
        st.write(processed_text)
        word2vec_text = word_vector(processed_text, 400)
        
        word2vec_prediction = svm_word2vec_best.predict(word2vec_text)[0]

        st.write("Word2Vec dengan Support Vector Machine:")
        st.write(f"Sentimen: {word2vec_prediction}")
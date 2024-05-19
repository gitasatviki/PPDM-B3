# -*- coding: utf-8 -*-
"""PPDM Kelompok B3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jgiGSwRY3AHZVzIRDaESinlDaDvUq9_A

**DOWNLOAD LIBRARY**
"""

!pip install Sastrawi

!pip install swifter

!pip install wordcloud

"""**LIBRARY YANG DIGUNAKAN**"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import string
import gensim
import pickle
from pathlib import Path
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

nltk.download('punkt')
nltk.download('stopwords')

"""**LIHAT DATA**"""

ulasan = pd.read_csv('bagi_sama_pinterest.csv')

positive_count = ulasan[ulasan['label'] == 'Positif'].shape[0]
negative_count = ulasan[ulasan['label'] == 'Negatif'].shape[0]

labels = ['Positif', 'Negatif']
counts = [positive_count, negative_count]

plt.bar(labels, counts, color=['green', 'red'])
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.show()

"""**CEK NULL**"""

ulasan.isnull().sum()

"""**CLEANING DATA**"""

def cleaning_ulasan(ulasan):
    ulasan = re.sub(r'@[A-Za-a0-9]+', ' ', ulasan)
    ulasan = re.sub(r'#[A-Za-z0-9]+', ' ', ulasan)
    ulasan = re.sub(r"http\S+", ' ', ulasan)
    ulasan = re.sub(r'[0-9]+', ' ', ulasan)
    ulasan = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", ulasan)
    ulasan = ulasan.strip(' ')
    return ulasan
ulasan['cleaning'] = ulasan['review'].apply(cleaning_ulasan)

def clear_emoji(ulasan):
    return ulasan.encode('ascii', 'ignore').decode('ascii')
ulasan['hapusEmoji'] = ulasan['cleaning'].apply(clear_emoji)

def double(ulasan):
    pattern = re.compile(r'(.)\1{2,}', re.DOTALL)
    return pattern.sub(r'\1', ulasan)
ulasan['double'] = ulasan['hapusEmoji'].apply(double)
ulasan

"""**CASEFOLDING**"""

def case_folding_text(ulasan):
    return ulasan.lower()
ulasan['caseFolding'] = ulasan['double'].apply(case_folding_text)
ulasan['caseFolding']

"""**TOKENIZING**"""

def tokenizing_text(ulasan):
    return word_tokenize(ulasan)
ulasan['tokenizing'] = ulasan['caseFolding'].apply(tokenizing_text)
ulasan['tokenizing']

"""**FORMALISASI**"""

def convert_to_slangword(ulasan):
    kamus_slang = eval(open("kata_gaul.txt").read())
    pattern = re.compile(r'\b(' + '|'.join(kamus_slang.keys()) + r')\b')
    content = []
    for kata in ulasan:
        filter_slang = pattern.sub(lambda x: kamus_slang[x.group()], kata)
        content.append(filter_slang.lower())
    ulasan = content
    return ulasan
ulasan['formalisasi'] = ulasan['tokenizing'].apply(convert_to_slangword)
ulasan['formalisasi']

"""**STOPWORD REMOVAL**"""

daftar_stopword = stopwords.words('indonesian')
daftar_stopword = set(daftar_stopword)
def stopword_text(words):
    return [word for word in words if word not in daftar_stopword]

ulasan['stopwordRemoval'] = ulasan['formalisasi'].apply(stopword_text)
ulasan['stopwordRemoval']

"""**STEMMING**"""

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemmed_wrapper(term):
    return stemmer.stem(term)
term_dict = {}

for document in ulasan['stopwordRemoval']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ''

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)

def stemming_text(document):
    return [term_dict[term] for term in document]
ulasan['stemming'] = ulasan['stopwordRemoval'].apply(stemming_text)
ulasan['stemming']

"""**HASIL SELURUH PREPROCESSING**"""

ulasan

"""**BAGI DATA**"""

X = ulasan['stemming']
Y = ulasan['label']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=64)

x_train_text = [' '.join(tokens) for tokens in x_train]
x_test_text = [' '.join(tokens) for tokens in x_test]

"""**TF-IDF**"""

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, max_features=2000)
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train_text)
x_test_tfidf = tfidf_vectorizer.transform(x_test_text)

print("Panjang x_train_tfidf: ", x_train_tfidf.shape)
print("Panjang x_test_tfidf: ", x_test_tfidf.shape)

feature_names = tfidf_vectorizer.get_feature_names_out()

x_train_tfidf_df = pd.DataFrame(x_train_tfidf.toarray(), columns=feature_names)
print("TF-IDF untuk x_train:")
print(x_train_tfidf_df)

x_test_tfidf_df = pd.DataFrame(x_test_tfidf.toarray(), columns=feature_names)
print("TF-IDF untuk x_test:")
print(x_test_tfidf_df)

"""**WORD2VEC**"""

model_w2v = gensim.models.Word2Vec(
    sentences=ulasan['stemming'],
    vector_size=400,
    window=5,
    min_count=2,
    sg=1,
    hs=0,
    negative=10,
    workers=2,
    seed=34,
    epochs=20,
    alpha=0.04,
    min_alpha=0.0006
)

model_w2v.train(ulasan['stemming'], total_examples=len(ulasan['stemming']), epochs=20)

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

x_train_word2vec = np.concatenate([word_vector(tokens, 400) for tokens in x_train])
x_test_word2vec = np.concatenate([word_vector(tokens, 400) for tokens in x_test])

print("Dimensi x_train_word2vec: ", x_train_word2vec.shape)
print("Dimensi x_test_word2vec: ", x_test_word2vec.shape)

np.save('x_train_word2vec.npy', x_train_word2vec)
np.save('x_test_word2vec.npy', x_test_word2vec)

"""**SVM TF-IDF DENGAN K-FOLD CROSS VALIDATION**"""

param_grid = {
    'C': [0.01, 0.05, 0.25, 0.5, 0.75, 1],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

sv_tfidf = SVC()
grid_search = GridSearchCV(sv_tfidf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train_tfidf, y_train)

best_params = grid_search.best_params_
best_accuracy_tfidf = grid_search.best_score_
print(f"Parameter terbaik: {best_params}")
print(f"Akurasi terbaik: {best_accuracy_tfidf}")

sv_tfidf_best = SVC(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'])
sv_tfidf_best.fit(x_train_tfidf, y_train)

print("\nClassification Report for TF-IDF with Best Parameters:")
print(classification_report(y_test, sv_tfidf_best.predict(x_test_tfidf)))

"""**SVM WORD2VEC DENGAN K-FOLD CROSS VALIDATION**"""

param_grid = {
    'C': [0.01, 0.05, 0.25, 0.5, 0.75, 1],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}

sv_word2vec = SVC()
grid_search = GridSearchCV(sv_word2vec, param_grid, cv=10, scoring='accuracy')
grid_search.fit(x_train_word2vec, y_train)

best_params = grid_search.best_params_
best_accuracy_word2vec = grid_search.best_score_
print(f"Parameter terbaik: {best_params}")
print(f"Akurasi terbaik: {best_accuracy_word2vec}")

sv_word2vec_best = SVC(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'])
sv_word2vec_best.fit(x_train_word2vec, y_train)

print("\nClassification Report for Word2Vec with Best Parameters:")
print(classification_report(y_test, sv_word2vec_best.predict(x_test_word2vec)))

"""**UBAH KE PICKLE**"""

with open('model_word2vec.pkl', 'wb') as f:
    pickle.dump(model_w2v, f)

with open('model_svm_word2vec.pkl', 'wb') as f:
    pickle.dump(sv_word2vec_best, f)
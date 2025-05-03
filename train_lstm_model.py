import pandas as pd
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# --- Setup awal ---
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)  # hapus simbol dan angka
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    hasil = stemmer.stem(' '.join(tokens))
    return hasil

# Studi Kasus: Kenaikan PPN 12%
# --- Baca data Komentar Kenaikan PPN 12% ---
df = pd.read_csv('dataset_tiktok_sentiment.csv')
print("Jumlah komentar studi kasus Kenaikan PPN 12%:", len(df))

# --- Preprocessing ---
df['clean_komentar'] = df['komentar'].apply(preprocess)
print("Contoh hasil preprocessing komentar (Kenaikan PPN 12%):\n", df[['komentar', 'clean_komentar']].head())

# --- Tokenisasi dan Padding ---
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['clean_komentar'])
sequences = tokenizer.texts_to_sequences(df['clean_komentar'])

max_len = 100
X = pad_sequences(sequences, maxlen=max_len)

# --- Encode Label ---
encoder = LabelEncoder()
y = encoder.fit_transform(df['sentiment'])
num_classes = len(encoder.classes_)

# --- Simpan label encoder & tokenizer ---
os.makedirs("model", exist_ok=True)
with open('model/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
with open('model/label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# --- Bangun Model LSTM ---
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128, input_length=max_len))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# --- Training Model ---
model.fit(X, y, epochs=5, batch_size=64, validation_split=0.2)

# --- Evaluasi ---
y_pred = model.predict(X)
y_pred_labels = np.argmax(y_pred, axis=1)
print("\nConfusion Matrix:")
print(confusion_matrix(y, y_pred_labels))
print("\nClassification Report:")
print(classification_report(y, y_pred_labels, target_names=encoder.classes_))

# --- Simpan Model ---
model.save('model/lstm_model.h5')
print("\nModel dan tokenizer berhasil disimpan di folder 'model/'")

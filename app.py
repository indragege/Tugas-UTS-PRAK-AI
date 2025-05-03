from flask import Flask, render_template, request
import pickle
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import nltk
import pandas as pd  # untuk menghitung data visualisasi
import os
nltk.download('stopwords')

# --- Inisialisasi
app = Flask(__name__)
model = load_model('model/lstm_model.h5')

# Load tokenizer dan label encoder
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

stop_words = set(stopwords.words('indonesian'))
stemmer = StemmerFactory().create_stemmer()
max_len = 100

# Studi Kasus: Kenaikan PPN 12%
# --- Data & Metrik Komentar Kenaikan PPN 12% ---
df = pd.read_csv('dataset_tiktok_sentiment.csv', dtype=str)
# Tambahkan kolom platform default jika tidak ada
if 'platform' not in df.columns:
    df['platform'] = 'tiktok'
# Konversi kolom timestamp ke datetime
df['time'] = pd.to_datetime(df['timestamp'], errors='coerce')
total_comments = len(df)
print("Total komentar studi kasus Kenaikan PPN 12%:", total_comments)
sentiment_counts = df['sentiment'].value_counts().to_dict()
# Format tanggal sebagai string 'YYYY-MM-DD' agar kompatibel dengan JSON
df['date'] = df['time'].dt.strftime('%Y-%m-%d')
df['hour'] = df['time'].dt.hour
comments_per_day = df['date'].value_counts().sort_index().to_dict()
comments_per_hour = df['hour'].value_counts().sort_index().to_dict()
comments_per_platform = df['platform'].value_counts().to_dict()

# --- Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    stemmed = stemmer.stem(' '.join(words))
    return stemmed

# --- Routes
@app.route('/')
def home():
    return render_template('index.html',
        total_comments=total_comments,
        sentiment_counts=sentiment_counts,
        comments_per_day=comments_per_day,
        comments_per_hour=comments_per_hour,
        comments_per_platform=comments_per_platform
    )

@app.route('/predict', methods=['POST'])
def predict():
    komentar = request.form['komentar']
    clean_text = preprocess(komentar)
    seq = tokenizer.texts_to_sequences([clean_text])
    pad = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(pad)
    label = label_encoder.inverse_transform([np.argmax(pred)])
    return render_template('index.html', komentar=komentar, label=label[0],
        total_comments=total_comments,
        sentiment_counts=sentiment_counts,
        comments_per_day=comments_per_day,
        comments_per_hour=comments_per_hour,
        comments_per_platform=comments_per_platform
    )

# --- Run
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

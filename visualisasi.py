import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load data
df = pd.read_csv("dataset_tiktok_sentiment.csv", dtype=str)

# Tambahkan kolom platform default jika tidak ada
if 'platform' not in df.columns:
    df['platform'] = 'tiktok'

# Konversi kolom timestamp_readable ke datetime
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Visualisasi 1: Distribusi Sentimen
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='sentiment', palette='Set2')
plt.title("Distribusi Sentimen Kenaikan PPN 12%")
plt.xlabel("Sentimen")
plt.ylabel("Jumlah Komentar")
plt.tight_layout()
plt.savefig("static/sentimen_distribusi.png")
plt.close()

# Visualisasi 2: Aktivitas Komentar per Hari
df['tanggal'] = df['timestamp'].dt.date
plt.figure(figsize=(10, 5))
df.groupby('tanggal').size().plot()
plt.title("Jumlah Komentar per Hari - Kenaikan PPN 12%")
plt.xlabel("Tanggal")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.savefig("static/komentar_per_hari.png")
plt.close()

# Visualisasi 3: Aktivitas Komentar per Platform
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='platform', palette='pastel')
plt.title("Komentar Berdasarkan Platform - Kenaikan PPN 12%")
plt.xlabel("Platform")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.savefig("static/komentar_per_platform.png")
plt.close()

# Preprocessing komentar
df['cleaned'] = df['komentar'].astype(str).str.lower()

# Wordcloud per kategori sentimen
for label in df['sentiment'].unique():
    teks = " ".join(df[df['sentiment'] == label]['cleaned'])
    wc = WordCloud(width=800, height=800, background_color='white').generate(teks)
    plt.figure(figsize=(6, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Wordcloud - {label}")
    plt.tight_layout()
    plt.savefig(f"static/wordcloud_{label.lower()}.png")
    plt.close()

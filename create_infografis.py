import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

# Membuat folder static jika belum ada
ostatic_dir = os.path.join(os.getcwd(), 'static')
if not os.path.exists(ostatic_dir):
    os.makedirs(ostatic_dir)

# Baca data
csv_path = 'dataset_tiktok_sentiment.csv'
df = pd.read_csv(csv_path, dtype=str)
# Tambahkan kolom platform default jika belum ada
if 'platform' not in df.columns:
    df['platform'] = 'tiktok'
# Konversi kolom timestamp jika ada
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Hitung metrik
sentiment_counts = df['sentiment'].value_counts()
platform_counts = df['platform'].value_counts()

def generate_sentiment_chart(output_path='static/infografis_sentimen.png'):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    colors = ['#28a745' if label=='positif' else '#dc3545' if label=='negatif' else '#6c757d' for label in sentiment_counts.index]
    sentiment_counts.plot(kind='bar', color=colors, ax=ax)
    ax.set_title('Distribusi Sentimen Kenaikan PPN 12%')
    ax.set_ylabel('Jumlah')
    ax.set_xlabel('Sentimen')
    ax.tick_params(axis='x', rotation=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Infografis Sentimen disimpan di: {output_path}')

def generate_platform_chart(output_path='static/infografis_platform.png'):
    fig, ax = plt.subplots(figsize=(6, 6), facecolor='white')
    platform_counts.plot(kind='bar', color=plt.get_cmap('tab10').colors[:len(platform_counts)], ax=ax)
    ax.set_title('Komentar per Platform - Kenaikan PPN 12%')
    ax.set_ylabel('Jumlah')
    ax.set_xlabel('Platform')
    ax.tick_params(axis='x', rotation=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Infografis Platform disimpan di: {output_path}')

def generate_wordcloud_images():
    wc_files = [
        ('static/infografis_wordcloud_positif.png', 'static/wordcloud_positif.png'),
        ('static/infografis_wordcloud_negatif.png', 'static/wordcloud_negatif.png'),
        ('static/infografis_wordcloud_netral.png', 'static/wordcloud_netral.png')
    ]
    for dst, src in wc_files:
        if os.path.exists(src):
            img = Image.open(src)
            img.save(dst)
            print(f'Wordcloud disimpan di: {dst}')

if __name__ == '__main__':
    generate_sentiment_chart()
    generate_platform_chart()
    generate_wordcloud_images() 
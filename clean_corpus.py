import pandas as pd
import re

DATELINE = re.compile(r"^\s*[A-Za-z]+(?: [A-Za-z]+)?\s*(?:\(\s*antara\s*\)|,\s*(?:idn times|kompas\.com|kompas\.tv)|:)\s*[-–—]?\s*", re.I)

def clean_corpus(input_file, output_file):
    print("Memproses pembersihan korpus...")
    
    # 1. Baca hanya sheet pertama, abaikan sheet 'bm25'
    df = pd.read_excel(input_file, sheet_name=0)
    # Ubah nama kolom menjadi lowercase setelah DataFrame (df) terdefinisi
    df.columns = [str(c).strip().lower() for c in df.columns]
    
    # 2. Hapus kolom catatan (fahmi, reza, Unnamed)
    kolom_inti = ['doc_id', 'title', 'media', 'topic', 'date', 'tanggal', 'url', 'text']
    df = df[[c for c in kolom_inti if c in df.columns]].copy()
    
    # 3. Hapus duplikat berdasarkan URL dan Judul (menyelesaikan isu D049/D050 vs D068/D069)
    df.drop_duplicates(subset=['url'], keep='first', inplace=True)
    df.drop_duplicates(subset=['title'], keep='first', inplace=True)
    
    # 4. Satukan label topik (menghilangkan redundansi)
    df['topic'] = df['topic'].str.strip().str.title()
    df['topic'] = df['topic'].replace({"E-Sport": "E-Sports"})
    df['media'] = df['media'].astype(str).str.strip().map(lambda m: m.title() if m.islower() else m)
    
    # 5. Perbaiki URL D007 yang relatif
    df['url'] = df['url'].apply(lambda x: f"https://{x}" if isinstance(x, str) and not x.startswith('http') else x)
    
    # 6. Hapus boilerplate "Baca juga: <judul>" dari dokumen Antara/Kompas
    def remove_baca_juga(text):
        if not isinstance(text, str): return text
        # Menghapus frasa "Baca juga:" sampai akhir baris
        return re.sub(r"baca juga:.*?(?=\n|$)", " ", text, flags=re.IGNORECASE)

    df['text'] = df['text'].apply(remove_baca_juga)
    
    # 7. Menghapus dateline dari teks (dikeluarkan dari fungsi remove_baca_juga)
    df['text'] = df['text'].apply(lambda t: DATELINE.sub("", t, count=1) if isinstance(t, str) else t)
    
    # 8. Filter teks dengan panjang minimal 15 kata (dikeluarkan dari fungsi remove_baca_juga)
    df = df[df['text'].str.split().str.len() >= 15].reset_index(drop=True)
    
    # Simpan hasil bersih
    df.to_excel(output_file, index=False)
    print(f"Korpus bersih tersimpan di: {output_file}")
    print(f"Total dokumen: {len(df)}")

if __name__ == "__main__":
    # Sesuaikan dengan nama file asli di folder Anda
    clean_corpus("Korpus_Gabungan_Timnas_Indonesia (3).xlsx", "Korpus_Bersih.xlsx")
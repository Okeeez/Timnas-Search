import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from functools import lru_cache

# Mempertahankan angka penting dan singkatan
KATA_PENTING = {"cup","vs","fc","tim","gol","sub","sty","afc","aff","u23","u22","u20","u19","u17","gbk","pssi", "3x3", "sea", "wni"}
# Mencegah over-stemming
PROTECTED = {"asian", "asean", "ketum"}

KATA_NEGASI  = {"tidak", "bukan", "jangan", "tanpa", "belum"}
STOPWORDS = set(StopWordRemoverFactory().get_stop_words()) - KATA_NEGASI

stemmer = StemmerFactory().create_stemmer()
NOISE_MEDIA = r"(?:[a-z]+,\s*)?(?:idn times|kompas\.?\s?tv|antara(?: news)?|detik\w*|cnn indonesia)\s*[-–—]\s*"

@lru_cache(maxsize=None)
def stem_token(t):
    t = t.strip(".-")
    if t in PROTECTED or t in KATA_PENTING:
        return t
    return stemmer.stem(t) if t.isalpha() else t

def preprocess(text):
    t = str(text).lower().replace("\xa0", " ")
    # Hapus dateline media
    t = re.sub(NOISE_MEDIA, "", t)
    # Satukan variasi u-20 menjadi u20 (sebelum tokenisasi)
    t = re.sub(r"\bu-(\d{2})\b", r"u\1", t)
    # Buang tanda baca
    t = re.sub(r"[\"“”‘’':;!?,()\[\]]", " ", t)
    
    toks = [x.strip(".-") for x in re.findall(r"[a-z0-9_.\-]+", t)]
    # Filter panjang token (kecuali angka atau kata penting)
    toks = [x for x in toks if x and (re.fullmatch(r"\d+(?:[.\-]\d+)*", x) or len(x) >= 4 or x in KATA_PENTING)]
    
    return [stem_token(x) for x in toks if x not in STOPWORDS]
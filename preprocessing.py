# preprocessing.py
import os
import re
from functools import lru_cache
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROTECTED_FILE = os.path.join(BASE_DIR, "data", "protected_words.txt")

# Mempertahankan angka penting dan singkatan domain olahraga/timnas
KATA_PENTING = {
    "cup", "vs", "fc", "tim", "gol", "sub", "sty", "afc", "aff",
    "u23", "u22", "u20", "u19", "u17", "gbk", "pssi", "3x3", "sea", "wni",
    "run", "ibl", "iss", "npc", "bri", "ea", "tc"
}

DEFAULT_PROTECTED = {"asian", "asean", "ketum", "nusi", "perbasi", "kediri", "bali", "oman"}


def load_protected_words(filepath: str = PROTECTED_FILE) -> set:
    words = set(DEFAULT_PROTECTED)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                w = line.strip().lower()
                if w and not w.startswith("#"):
                    words.add(w)
    return words


PROTECTED = load_protected_words()

KATA_NEGASI = {"tidak", "bukan", "jangan", "tanpa", "belum"}
STOPWORDS = set(StopWordRemoverFactory().get_stop_words()) - KATA_NEGASI

stemmer = StemmerFactory().create_stemmer()
NOISE_MEDIA = r"(?:[a-z]+,\s*)?(?:idn times|kompas\.?\s?tv|antara(?: news)?|detik\w*|cnn indonesia)\s*[-–—]\s*"


@lru_cache(maxsize=None)
def stem_token(t: str) -> str:
    t = t.strip(".-")
    if t in PROTECTED or t in KATA_PENTING:
        return t
    return stemmer.stem(t) if t.isalpha() else t


def preprocess(text: str) -> list:
    t = str(text).lower().replace("\xa0", " ")
    # Hapus dateline media
    t = re.sub(NOISE_MEDIA, "", t)
    # Satukan variasi u-20 menjadi u20 dan varian e-sport(s)/esport(s) menjadi esport sebelum tokenisasi
    t = re.sub(r"\bu-(\d{2})\b", r"u\1", t)
    t = re.sub(r"\be-?sports?\b", "esport", t)
    # Buang tanda baca
    t = re.sub(r"[\"“”‘’':;!?,()\[\]]", " ", t)

    toks = [x.strip(".-") for x in re.findall(r"[a-z0-9_.\-]+", t)]
    # Filter panjang token (kecuali angka atau kata penting)
    toks = [
        x for x in toks
        if x and (re.fullmatch(r"\d+(?:[.\-]\d+)*", x) or len(x) >= 4 or x in KATA_PENTING)
    ]

    return [stem_token(x) for x in toks if x not in STOPWORDS]

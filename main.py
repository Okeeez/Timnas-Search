# main.py
import os, time, math, base64, html, hashlib, re, shutil
from functools import lru_cache
import numpy as np, pandas as pd
import gradio as gr
from collections import defaultdict, Counter

# Cek versi Gradio di awal sebelum proses berat dijalankan
def _gradio_major_version(ver_str: str) -> int:
    m = re.match(r"^\s*(\d+)", str(ver_str))
    return int(m.group(1)) if m else 0

GRADIO_VER = getattr(gr, "__version__", "0.0.0")
if _gradio_major_version(GRADIO_VER) < 6:
    print(f"ERROR: Aplikasi ini membutuhkan gradio>=6.0, versi terpasang: {GRADIO_VER}")
    raise SystemExit(1)

# Direktori tempat main.py berada agar file data, cache, dan embedding konsisten di lokal & Linux
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Konfigurasi cache HuggingFace: default D:\hf_cache di Windows, atau <BASE_DIR>/hf_cache di Linux/server
if os.name == "nt":
    DEFAULT_HF_CACHE = r"D:\hf_cache"
else:
    DEFAULT_HF_CACHE = os.path.join(BASE_DIR, "hf_cache")

HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", DEFAULT_HF_CACHE)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.makedirs(HF_CACHE_DIR, exist_ok=True)

from preprocessing import preprocess, stem_token  # Import dari file terpisah

# ===== KONFIGURASI =====
EMBED_MODEL  = "intfloat/multilingual-e5-small"
# Alternatif lain misalnya: "paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
USE_RERANK   = False

CHUNK_WORDS, CHUNK_STRIDE = 80, 50
TITLE_WEIGHT = 0.5  # Bobot skor judul saat dikombinasikan dengan skor chunk terbaik
SEM_MIN      = 0.72 # Batas mutlak minimum skor semantik (sesuaikan jika memakai model non-e5)
SEM_REL      = 0.85 # Batas relatif terhadap skor semantik tertinggi (skor >= SEM_REL * max_sem)
SEM_TOP_K    = 30   # Batas maksimum jumlah dokumen hasil semantik

HYBRID_K     = 60   # Konstanta redaman Reciprocal Rank Fusion (RRF)
HYBRID_W_LEX = 1.0  # Bobot kontribusi peringkat leksikal (BM25) pada Hybrid RRF
HYBRID_W_SEM = 1.0  # Bobot kontribusi peringkat semantik pada Hybrid RRF

FILE_CORPUS = os.path.join(BASE_DIR, "Korpus_Bersih.xlsx")  # Gunakan korpus yang sudah dibersihkan
FILE_QRELS  = os.path.join(BASE_DIR, "qrels.csv")
FILE_QRELS_TEMPLATE = os.path.join(BASE_DIR, "qrels_template.csv")
DOSEN       = "Dr. Nurul Chamidah, S.Kom., M.Kom."
TIM = [("Fahmi Idrus", "2310511079"), ("Rezaul Karim", "2310511134"), ("Rayhan Andika Putra", "2310511122")]
PER_PAGE    = 10

# Memuat Korpus
try:
    df = pd.read_excel(FILE_CORPUS)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ["doc_id", "title", "media", "topic", "url", "text"]: df[c] = df[c].fillna("").astype(str)
    if "tanggal" in df.columns and "date" not in df.columns:
        df.rename(columns={"tanggal": "date"}, inplace=True)
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT # Jika benar-benar tidak ada kolom tanggal
    print(f"{len(df)} dokumen termuat dari {FILE_CORPUS}.")
except FileNotFoundError:
    print(f"ERROR: File {FILE_CORPUS} tidak ditemukan. Jalankan clean_corpus.py terlebih dahulu.")
    exit(1)

# Indexing BM25 & TF-IDF
class Index:
    def __init__(self, docs):
        self.N = len(docs); self.dl = np.array([len(d) for d in docs], float); self.avgdl = self.dl.mean() or 1.0
        self.post = defaultdict(dict)
        for i, toks in enumerate(docs):
            for t, c in Counter(toks).items(): self.post[t][i] = c
        self.idf_bm = {t: math.log(1 + (self.N - len(p) + .5) / (len(p) + .5)) for t, p in self.post.items()}
        self.idf_tf = {t: math.log((1 + self.N) / (1 + len(p))) + 1 for t, p in self.post.items()}
        sq = np.zeros(self.N)
        for t, p in self.post.items():
            for i, tf in p.items(): sq[i] += ((1 + math.log(tf)) * self.idf_tf[t]) ** 2
        self.norm = np.sqrt(sq)

    def bm25(self, terms, k1=1.5, b=0.75):
        s = np.zeros(self.N)
        for t in set(terms):
            for i, tf in self.post.get(t, {}).items():
                s[i] += self.idf_bm[t] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.dl[i] / self.avgdl))
        return s

    def tfidf(self, terms):
        s, qn = np.zeros(self.N), 0.0
        for t, c in Counter(terms).items():
            if t not in self.post: continue
            wq = (1 + math.log(c)) * self.idf_tf[t]; qn += wq * wq
            for i, tf in self.post[t].items(): s[i] += wq * (1 + math.log(tf)) * self.idf_tf[t]
        return s / (math.sqrt(qn) * self.norm + 1e-12) if qn else s

df["tokens"] = (df["title"] + " " + df["text"]).apply(preprocess)
INDEX = Index(df["tokens"].tolist())
if INDEX.N != len(df):
    print(f"⚠️ PERINGATAN: Jumlah dokumen tidak konsisten! df={len(df)} dokumen, INDEX.N={INDEX.N} dokumen.")
print(f"Index siap: {len(df)} dokumen termuat (df), {INDEX.N} dokumen terindeks (INDEX.N), {len(INDEX.post)} term unik.")

# Helper prefix untuk model keluarga E5
def _is_e5(model_name: str) -> bool:
    return "e5" in str(model_name).lower()

def _prep_query_text(query: str) -> str:
    q = str(query).strip()
    return f"query: {q}" if _is_e5(EMBED_MODEL) else q

def _prep_passage_text(text: str) -> str:
    p = str(text).strip()
    return f"passage: {p}" if _is_e5(EMBED_MODEL) else p

# Semantic Index (Chunk Embedding + Title Embedding)
CHUNKS = []
TITLES_RAW = []
for i, r in df.iterrows():
    TITLES_RAW.append(r["title"])
    w = r["text"].split() or [r["title"]]
    for st in range(0, len(w), CHUNK_STRIDE):
        seg = " ".join(w[st:st + CHUNK_WORDS])
        CHUNKS.append((i, seg, f'{r["title"]}. {seg}'))
        if st + CHUNK_WORDS >= len(w): break
CH_DOC   = np.array([c[0] for c in CHUNKS])
CH_START = np.r_[0, np.flatnonzero(np.diff(CH_DOC)) + 1]

emb_status_detail = "belum diproses"

# Jika unduhan gagal karena cache parsial korup di C:, hapus folder:
# C:\Users\rayha\.cache\huggingface\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2
try:
    from sentence_transformers import SentenceTransformer
    chunk_payload = "|".join(c[2] for c in CHUNKS)
    title_payload = "|".join(TITLES_RAW)
    chunk_key = hashlib.md5(f"{EMBED_MODEL}|chunks|{CHUNK_WORDS}|{CHUNK_STRIDE}|{chunk_payload}".encode()).hexdigest()[:10]
    title_key = hashlib.md5(f"{EMBED_MODEL}|titles|{title_payload}".encode()).hexdigest()[:10]
    emb_file = os.path.join(BASE_DIR, f"emb_{chunk_key}.npy")
    title_emb_file = os.path.join(BASE_DIR, f"emb_title_{title_key}.npy")

    chunk_exists = os.path.exists(emb_file)
    title_exists = os.path.exists(title_emb_file)

    SEM_MODEL = SentenceTransformer(EMBED_MODEL, cache_folder=HF_CACHE_DIR)
    if not chunk_exists or not title_exists:
        print(f"Mengunduh & memproses embedding semantic ({EMBED_MODEL})...")

    if chunk_exists:
        CH_EMB = np.load(emb_file)
    else:
        passage_chunks = [_prep_passage_text(c[2]) for c in CHUNKS]
        CH_EMB = SEM_MODEL.encode(passage_chunks, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        np.save(emb_file, CH_EMB)

    if title_exists:
        TITLE_EMB = np.load(title_emb_file)
    else:
        passage_titles = [_prep_passage_text(t) for t in TITLES_RAW]
        TITLE_EMB = SEM_MODEL.encode(passage_titles, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        np.save(title_emb_file, TITLE_EMB)

    if chunk_exists and title_exists:
        emb_status_detail = f"file embedding ditemukan ({os.path.basename(emb_file)}, {os.path.basename(title_emb_file)})"
    elif not chunk_exists and not title_exists:
        emb_status_detail = f"di-encode ulang seluruhnya ({os.path.basename(emb_file)}, {os.path.basename(title_emb_file)})"
    else:
        emb_status_detail = f"di-encode ulang sebagian (chunks={'ditemukan' if chunk_exists else 'baru'}, titles={'ditemukan' if title_exists else 'baru'})"

    SEM_OK = True
    print(f"Semantic index siap: {len(CHUNKS)} potongan, {len(TITLES_RAW)} judul.")
except Exception as e:
    SEM_OK, SEM_MODEL, CH_EMB, TITLE_EMB = False, None, None, None
    emb_status_detail = f"gagal dimuat ({e})"
    try:
        free_mb = shutil.disk_usage(HF_CACHE_DIR).free / (1024 * 1024)
        disk_info = f"Sisa ruang disk di '{HF_CACHE_DIR}': {free_mb:.2f} MB."
    except Exception:
        disk_info = f"Tidak dapat memeriksa sisa ruang disk di '{HF_CACHE_DIR}'."
    print(
        f"⚠️ Semantic search tidak aktif: {e}\n"
        f"   {disk_info} Pastikan ruang kosong memadai (minimal ~600 MB untuk model & cache) "
        f"atau ubah HF_CACHE_DIR / HF_HOME ke drive lain yang memiliki ruang cukup. Aplikasi tetap berjalan dengan BM25."
    )

print(
    f"[STARTUP] Gradio={GRADIO_VER} | SEM_OK={SEM_OK} | EMBED_MODEL={EMBED_MODEL} "
    f"| HF_CACHE_DIR={HF_CACHE_DIR} | Embedding={emb_status_detail}"
)

RERANKER = None
if SEM_OK and USE_RERANK:
    try:
        from sentence_transformers import CrossEncoder
        RERANKER = CrossEncoder(RERANK_MODEL, cache_folder=HF_CACHE_DIR)
        print(f"Cross-encoder reranker siap: {RERANK_MODEL}")
    except Exception as e:
        RERANKER = None
        print(f"⚠️ Reranker tidak dimuat (dilewati): {e}")

@lru_cache(maxsize=512)
def _encode_query_cached(query_text: str, model_name: str):
    q_formatted = _prep_query_text(query_text)
    vec = SEM_MODEL.encode([q_formatted], batch_size=1, normalize_embeddings=True, show_progress_bar=False)[0]
    return vec.tobytes(), vec.shape, str(vec.dtype)

def encode_query(query: str) -> np.ndarray:
    raw, shape, dtype = _encode_query_cached(str(query), EMBED_MODEL)
    return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape)

def semantic_scores(query):
    q = encode_query(query)
    sims = CH_EMB @ q
    chunk_best = np.maximum.reduceat(sims, CH_START)
    title_sims = TITLE_EMB @ q
    combined = np.maximum(chunk_best, TITLE_WEIGHT * title_sims + (1.0 - TITLE_WEIGHT) * chunk_best)
    return combined, sims

def filter_semantic_candidates(sem_scores: np.ndarray, mask: np.ndarray):
    """Kembalikan array indeks dokumen yang lolos threshold adaptif (SEM_MIN & SEM_REL) dan SEM_TOP_K."""
    valid = np.where(mask & (sem_scores >= SEM_MIN))[0]
    if len(valid) == 0:
        return np.array([], dtype=int)
    max_sc = float(np.max(sem_scores[valid]))
    rel_cut = SEM_REL * max_sc
    keep = valid[sem_scores[valid] >= rel_cut]
    keep = keep[np.argsort(-sem_scores[keep], kind="stable")]
    if SEM_TOP_K:
        keep = keep[:SEM_TOP_K]
    return keep

def best_chunk(i, sims):
    s = CH_START[i]; e = CH_START[i + 1] if i + 1 < len(CH_START) else len(sims)
    return CHUNKS[s + int(np.argmax(sims[s:e]))][1]

def rerank_hits(query: str, hits: list, top_rerank: int = 20) -> list:
    if not RERANKER or len(hits) <= 1:
        return hits
    head = hits[:top_rerank]
    tail = hits[top_rerank:]
    pairs = []
    for doc_idx, _, ch in head:
        row = df.iloc[doc_idx]
        passage = ch if ch else f"{row['title']}. {row['text'][:300]}"
        pairs.append((query, passage))
    rr_scores = RERANKER.predict(pairs, batch_size=16, show_progress_bar=False)
    rescored = sorted(
        [(doc_idx, float(rr), ch) for (doc_idx, _, ch), rr in zip(head, rr_scores)],
        key=lambda x: -x[1]
    )
    return rescored + tail

def filter_mask(topik="Semua", waktu="Semua", media="Semua"):
    m = np.ones(len(df), bool)
    if topik != "Semua": m &= (df["topic"] == topik).values
    if media != "Semua": m &= (df["media"] == media).values
    if waktu != "Semua" and df["date"].notna().any():
        if waktu.startswith("Tahun"): m &= (df["date"].dt.year == int(waktu.split()[-1])).values
        else:
            days = {"7 hari terakhir": 7, "30 hari terakhir": 30, "1 tahun terakhir": 365}[waktu]
            m &= (df["date"] >= df["date"].max() - pd.Timedelta(days=days)).values
    return m

def search(query, model="BM25", topik="Semua", waktu="Semua", media="Semua", top_n=None, k1=1.5, b=0.75):
    t0 = time.perf_counter()
    if model in ("Semantic", "Hybrid") and not SEM_OK: model = "BM25"
    terms = preprocess(query)
    mask, n = filter_mask(topik, waktu, media), len(df)
    lex, sem, sims = np.zeros(n), np.zeros(n), None
    
    if model != "Semantic" and terms: lex = INDEX.tfidf(terms) if model == "TF-IDF" else INDEX.bm25(terms, k1, b)
    if model in ("Semantic", "Hybrid"): sem, sims = semantic_scores(query)
    
    if model == "Semantic":
        sem_keep = filter_semantic_candidates(sem, mask)
        sc = np.zeros(n)
        sc[sem_keep] = sem[sem_keep]
        idx = sem_keep
    elif model == "Hybrid":
        sem_keep = filter_semantic_candidates(sem, mask)
        lex_keep = np.where(mask & (lex > 0))[0]
        lex_keep = lex_keep[np.argsort(-lex[lex_keep], kind="stable")]
        sc = np.zeros(n)
        if len(lex_keep) > 0:
            sc[lex_keep] += HYBRID_W_LEX * (1.0 / (HYBRID_K + np.arange(1, len(lex_keep) + 1)))
        if len(sem_keep) > 0:
            sc[sem_keep] += HYBRID_W_SEM * (1.0 / (HYBRID_K + np.arange(1, len(sem_keep) + 1)))
        idx = np.where((sc > 0) & mask)[0]
        idx = idx[np.argsort(-sc[idx], kind="stable")]
    else:
        sc = lex
        idx = np.where((sc > 0) & mask)[0]
        idx = idx[np.argsort(-sc[idx], kind="stable")]

    if top_n: idx = idx[:top_n]
    hits = [(int(i), float(sc[i]), best_chunk(int(i), sims) if sims is not None else None) for i in idx]
    if model == "Hybrid" and RERANKER and hits:
        hits = rerank_hits(query, hits, top_rerank=20)
        if top_n: hits = hits[:top_n]
    return hits, terms, time.perf_counter() - t0

# Query Pengujian Seimbang (Fakta, Parafrase, Semantic, Semantic/sinonim)
SYSTEMS = {"TF-IDF (baseline)": dict(model="TF-IDF"),
           "BM25": dict(model="BM25")}
if SEM_OK: SYSTEMS.update({"Semantic": dict(model="Semantic"), "Hybrid (BM25 + Semantic)": dict(model="Hybrid")})

TEST_QUERIES = [
    "Jadwal Timnas U20 di kualifikasi piala asia",             # Fact / u20 regex test
    "Pemain diaspora yang baru dinaturalisasi",                # Semantic/sinonim test
    "Strategi pelatih Shin Tae-yong melawan Jepang",           # Entity
    "Hasil pertandingan basket 3x3 putri SEA Games",           # KATA_PENTING test (3x3, sea)
    "Upaya peningkatan kompetensi pelatih angkat besi",        # Semantic paraphrase
    "Dukungan suporter di stadion GBK",                        # Semantic/sinonim
    "Kualitas tim sepak bola putri Garuda Pertiwi",            # Fact
    "Persiapan atlet disabilitas menuju Paralimpiade",         # Semantic
    "Tanggapan Ketum PSSI mengenai prestasi E-Sports",         # Semantic/sinonim & Protected words
    "Daftar pemain yang dicoret dari skuad"                    # Paraphrase (pemecatan/dicoret)
]

def buat_template_qrels():
    rows = []
    for qi, q in enumerate(TEST_QUERIES, 1):
        pool = []
        seen = set()
        for cfg in SYSTEMS.values():
            for i, *_ in search(q, top_n=10, **cfg)[0]:
                if i not in seen:
                    seen.add(i)
                    pool.append(i)
        for i in pool:
            rows.append(dict(query_id=f"Q{qi}", query=q, doc_id=df.doc_id[i], title=df.title[i], relevan=""))
    pd.DataFrame(rows).to_csv(FILE_QRELS_TEMPLATE, index=False)
    print(f"{FILE_QRELS_TEMPLATE} berhasil dibuat. Isi kolom 'relevan' dengan 1 atau 0, lalu simpan sebagai qrels.csv.")

if __name__ == "__main__":
    if not os.path.exists(FILE_QRELS_TEMPLATE): buat_template_qrels()
    
    # ------------------ BLOK UI GRADIO ------------------
    BG_IMAGE = os.path.join(BASE_DIR, "suporter.jpg")

    def bg_rule():
        if os.path.exists(BG_IMAGE):
            with open(BG_IMAGE, "rb") as f: 
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"""
            body, .gradio-container {{ 
                background: url('data:image/jpeg;base64,{b64}') no-repeat center center fixed !important; 
                background-size: cover !important; 
            }}
            /* Overlay adaptif: Background akan tertutup warna putih semi-transparan di Light Mode 
               dan warna hitam semi-transparan di Dark Mode agar teks selalu terbaca */
            .gradio-container {{ 
                background-color: var(--background-fill-primary) !important; 
                background-blend-mode: overlay; 
                opacity: 0.95; 
            }}
            """
        return "body, .gradio-container { background: var(--background-fill-primary) !important; }\n"

    CSS_STATIC = """
    .gradio-container { max-width: 95% !important; border: none !important; }
    
    /* Header layout - Menggunakan variabel warna teks adaptif */
    .top-header { display: flex; align-items: center; gap: 15px; margin-bottom: 20px; flex-wrap: wrap; }
    .logo-img { height: 40px; font-weight: bold; font-size: 1.5em; color: var(--body-text-color); display: flex; align-items: center; gap: 10px; }
    .logo-img span { color: #f87171; } /* Merah logo tetap dipertahankan */
    .sem-notice { font-size: 0.85rem; color: var(--body-text-color-subdued); margin-top: -12px; margin-bottom: 14px; }
    
    /* Search Bar Custom */
    .search-box-container { flex-grow: 1; max-width: 600px; }
    .search-box-container textarea { border-radius: 25px !important; padding: 10px 20px !important; font-size: 1.1em !important; background: var(--background-fill-primary) !important; color: var(--body-text-color) !important; border: 1px solid var(--border-color-primary) !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important; }
    
    /* Pill Filters */
    .filter-row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; justify-content: center; margin-top: -10px; margin-bottom: 20px; }
    .pill-dropdown { min-width: 120px !important; }
    .pill-dropdown .form { border-radius: 20px !important; background: var(--background-fill-secondary) !important; border: 1px solid var(--border-color-primary) !important; color: var(--body-text-color) !important; text-align: center; }
    
    /* Google-style Results Adaptif */
    .g-result { margin-bottom: 25px; max-width: 700px; }
    .g-url { font-size: 0.85rem; color: var(--body-text-color-subdued); margin-bottom: 2px; display: flex; align-items: center; gap: 8px; }
    .g-url span { background: var(--background-fill-secondary); padding: 2px 6px; border-radius: 10px; font-size: 0.75rem; color: var(--body-text-color); }
    
    /* Judul biru, tapi dibuat lebih terang khusus di Dark Mode */
    .g-title { font-size: 1.3rem; color: #2563eb; text-decoration: none; font-weight: 500; display: block; line-height: 1.3; margin-bottom: 4px; }
    .dark .g-title { color: #60a5fa; } 
    .g-title:hover { text-decoration: underline; }
    
    .g-snip { font-size: 0.95rem; color: var(--body-text-color); line-height: 1.5; }
    .g-snip mark { background: rgba(96, 165, 250, 0.3); color: var(--body-text-color); font-weight: bold; border-radius: 3px; padding: 0 2px; }
    
    /* Tabs & Footer */
    .tabs { background: transparent !important; border: none !important; border-bottom: 1px solid var(--border-color-primary) !important; }
    .tab-nav button { color: var(--body-text-color-subdued) !important; font-weight: bold !important; text-transform: none !important; }
    .tab-nav button.selected { color: var(--body-text-color) !important; border-bottom: 3px solid #3b82f6 !important; }
    .stats-bar { font-size: 0.9rem; color: var(--body-text-color-subdued); margin-bottom: 20px; }
    
    .footer { display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--body-text-color-subdued); margin-top: 50px; padding-top: 10px; border-top: 1px solid var(--border-color-primary); }
    """

    def make_snippet(text, terms, chunk):
        if not text: return ""
        if chunk:
            src_text = chunk
        else:
            src_text = text[:350] + "..." if len(text) > 350 else text
        if not terms: return html.escape(src_text)
        
        esc = html.escape(src_text)
        for t in sorted(terms, key=len, reverse=True):
            if len(t) < 3: continue
            ptn = re.compile(f"({re.escape(html.escape(t))})", re.IGNORECASE)
            esc = ptn.sub(r"<mark>\1</mark>", esc)
        return esc

    def card_html(row, score, snippet):
        dt = row['date'].strftime('%d %b %Y') if pd.notnull(row.get('date')) else 'Tanpa Tanggal'
        url_display = str(row['url']).replace("https://", "").replace("http://", "").split("/")[0]
        
        # Desain ala Google di Mockup 2
        return f"""
        <div class="g-result">
            <div class="g-url">
                <img src="https://www.google.com/s2/favicons?domain={url_display}" style="width:16px;height:16px;background:white;border-radius:50%;" onerror="this.style.display='none'">
                {html.escape(str(row['media']))} • {url_display}
                <span>Skor: {score:.2f}</span>
            </div>
            <a href="{row['url']}" target="_blank" class="g-title">{html.escape(str(row['title']))}</a>
            <div class="g-snip">📅 {dt} — {snippet}</div>
        </div>
        """

    def hasil_html(hits, terms, dt, query, page=1):
        if not hits: return f"<div class='stats-bar'>Tidak ada hasil untuk <b>'{html.escape(query)}'</b>.</div>", gr.update(visible=False), gr.update(visible=False), gr.update(value=f"Halaman 1/1")
        
        total = len(hits)
        pages = math.ceil(total / PER_PAGE)
        page = max(1, min(page, pages))
        start = (page - 1) * PER_PAGE
        
        out = f"<div class='stats-bar'>{total} Jumlah Korpus ({dt:.2f} detik)</div>"
        
        for idx, score, chunk in hits[start:start+PER_PAGE]:
            row = df.iloc[idx]
            snip = make_snippet(row['text'], terms, chunk)
            out += card_html(row, score, snip)
            
        return out, gr.update(interactive=page>1, visible=True), gr.update(interactive=page<pages, visible=True), gr.update(value=f"Halaman {page}/{pages}", visible=True)

    def do_search(q, mod_label, top, wak, med):
        # Jika query kosong, batalkan pencarian
        if not q.strip(): 
            return "<div>Masukkan kata kunci.</div>", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), []
        
        # 1. BONGKAR DICTIONARY SYSTEMS
        # Ambil konfigurasi asli berdasarkan label yang dipilih di dropdown UI
        config = SYSTEMS.get(mod_label, {"model": "BM25"})
        actual_model = config["model"]
        
        # 2. PANGGIL FUNGSI SEARCH DENGAN PARAMETER YANG BENAR
        h, t, dt = search(q, model=actual_model, topik=top, waktu=wak, media=med)
        
        # Tampilkan hasil
        out, p, n, l = hasil_html(h, t, dt, q, 1)
        return out, p, n, l, {"hits": h, "terms": t, "dt": dt, "q": q, "page": 1}

    def change_page(state, dir_):
        if not state: return "<div>Error</div>", gr.update(), gr.update(), gr.update(), state
        state["page"] += dir_
        out, p, n, l = hasil_html(state["hits"], state["terms"], state["dt"], state["q"], state["page"])
        return out, p, n, l, state

    default_model_val = "Hybrid (BM25 + Semantic)" if SEM_OK else "BM25"

    with gr.Blocks(title="Timnas Search") as demo:
        state = gr.State({})
        
        # Header (Logo & Search Bar sejajar ala Mockup 2)
        with gr.Row(elem_classes="top-header"):
            gr.HTML("<div class='logo-img'>⚽ Timnas <span>Search</span></div>", scale=1)
            with gr.Column(scale=4, elem_classes="search-box-container"):
                q_in = gr.Textbox(placeholder="Ketik pencarian berita timnas...", show_label=False, container=False, lines=1)

        if not SEM_OK:
            gr.HTML("<div class='sem-notice'>Semantic search tidak aktif, memakai BM25</div>")
        
        # Pill Filters (Di bawah search bar)
        with gr.Row(elem_classes="filter-row"):
            top_in = gr.Dropdown(["Semua"] + sorted([str(x) for x in df['topic'].unique() if str(x).strip()]), value="Semua", label="Topik", container=False, elem_classes="pill-dropdown")
            wak_in = gr.Dropdown(["Semua", "7 hari terakhir", "30 hari terakhir", "1 tahun terakhir"], value="Semua", label="Waktu", container=False, elem_classes="pill-dropdown")
            med_in = gr.Dropdown(["Semua"] + sorted([str(x) for x in df['media'].unique() if str(x).strip()]), value="Semua", label="Media", container=False, elem_classes="pill-dropdown")
            mod_in = gr.Dropdown(list(SYSTEMS.keys()), value=default_model_val, label="Model (Korpus N)", container=False, elem_classes="pill-dropdown")
            btn = gr.Button("Search 🔍", variant="primary", elem_classes="pill-dropdown")

        # Tabs Layout
        with gr.Tabs(elem_classes="tabs"):
            with gr.TabItem("Search_Engine"):
                out_html = gr.HTML("<div style='min-height: 400px;'></div>")
                
                with gr.Row(elem_id="pagination", visible=False) as p_row:
                    prev_b = gr.Button("⬅️ Prev", visible=False)
                    page_l = gr.Label("Halaman 1/1", visible=False, container=False)
                    next_b = gr.Button("Next ➡️", visible=False)

            with gr.TabItem("Detail_Korpus"):
                gr.Markdown(f"### Total Korpus: {len(df)} Berita")
                gr.Dataframe(df[['date', 'title', 'media', 'topic']].head(50))
                
            with gr.TabItem("Tentang_Tim"):
                team = "".join(f"<li><b>{n}</b> ({i})</li>" for n, i in TIM)
                gr.HTML(f"<ul>{team}</ul>")

        # Footer
        gr.HTML(f"""
        <div class="footer">
            <div>Dosen: {DOSEN} | Mata Kuliah: Temu Kembali Informasi</div>
            <div>Credit By: Tim 4 | Tahun: 2026</div>
        </div>
        """)

        # Event Listeners
        btn.click(do_search, [q_in, mod_in, top_in, wak_in, med_in], [out_html, prev_b, next_b, page_l, state])
        q_in.submit(do_search, [q_in, mod_in, top_in, wak_in, med_in], [out_html, prev_b, next_b, page_l, state])
        prev_b.click(lambda s: change_page(s, -1), state, [out_html, prev_b, next_b, page_l, state])
        next_b.click(lambda s: change_page(s, 1), state, [out_html, prev_b, next_b, page_l, state])

    # Eksekusi dengan inject CSS, theme, dan background image script
    demo.launch(
        theme=gr.themes.Base(),
        css=bg_rule() + CSS_STATIC,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
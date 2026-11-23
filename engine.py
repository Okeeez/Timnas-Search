# engine.py
import os, time, math, hashlib, re, shutil
from functools import lru_cache
import numpy as np, pandas as pd
from collections import defaultdict, Counter

# Direktori tempat engine.py berada agar file data, cache, dan embedding konsisten di lokal & Linux
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Konfigurasi cache HuggingFace lintas platform dengan penanganan error
DEFAULT_HF_CACHE = os.path.join(BASE_DIR, "hf_cache")
HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", DEFAULT_HF_CACHE)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
try:
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
except OSError as e:
    print(f"PERINGATAN: Gagal membuat direktori cache HF '{HF_CACHE_DIR}': {e}")

from preprocessing import preprocess, stem_token  # Import dari file terpisah

# ===== KONFIGURASI =====
EMBED_MODEL  = "intfloat/multilingual-e5-small"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
USE_RERANK   = False

CHUNK_WORDS, CHUNK_STRIDE = 80, 50
TITLE_WEIGHT = 0.5  # Bobot skor judul saat dikombinasikan dengan skor chunk terbaik
SEM_Z        = 1.5  # Ambang z-score minimum skor semantik pada dokumen yang lolos mask
SEM_TOP_K    = 20   # Batas maksimum jumlah dokumen hasil semantik
LEX_MAX_DF   = 0.4  # Proporsi maksimum document frequency term pada cabang lexical Hybrid

HYBRID_K     = 60   # Konstanta redaman Reciprocal Rank Fusion (RRF)
HYBRID_W_LEX = 1.0  # Bobot kontribusi peringkat leksikal (BM25) pada Hybrid RRF
HYBRID_W_SEM = 1.0  # Bobot kontribusi peringkat semantik pada Hybrid RRF

FILE_CORPUS = os.path.join(BASE_DIR, "Korpus_Bersih.xlsx")
FILE_QUERIES = os.path.join(BASE_DIR, "data", "queries.csv")
FILE_QRELS  = os.path.join(BASE_DIR, "qrels.csv")
FILE_QRELS_TEMPLATE = os.path.join(BASE_DIR, "qrels_template.csv")
DOSEN       = "Dr. Nurul Chamidah, S.Kom., M.Kom."
TIM = [("Fahmi Idrus", "2310511079"), ("Rezaul Karim", "2310511134"), ("Rayhan Andika Putra", "2310511122")]
PER_PAGE    = 10

# Memuat Korpus
try:
    df = pd.read_excel(FILE_CORPUS)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in ["doc_id", "title", "media", "topic", "url", "text"]:
        df[c] = df[c].fillna("").astype(str)
    if "tanggal" in df.columns and "date" not in df.columns:
        df.rename(columns={"tanggal": "date"}, inplace=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        df["date"] = pd.NaT
    print(f"{len(df)} dokumen termuat dari {FILE_CORPUS}.")
except FileNotFoundError:
    print(f"ERROR: File {FILE_CORPUS} tidak ditemukan. Jalankan clean_corpus.py terlebih dahulu.")
    exit(1)

# Indexing BM25 & TF-IDF
class Index:
    def __init__(self, docs):
        self.N = len(docs)
        self.dl = np.array([len(d) for d in docs], float)
        self.avgdl = self.dl.mean() or 1.0
        self.post = defaultdict(dict)
        for i, toks in enumerate(docs):
            for t, c in Counter(toks).items():
                self.post[t][i] = c
        self.idf_bm = {t: math.log(1 + (self.N - len(p) + .5) / (len(p) + .5)) for t, p in self.post.items()}
        self.idf_tf = {t: math.log((1 + self.N) / (1 + len(p))) + 1 for t, p in self.post.items()}
        sq = np.zeros(self.N)
        for t, p in self.post.items():
            for i, tf in p.items():
                sq[i] += ((1 + math.log(tf)) * self.idf_tf[t]) ** 2
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
            if t not in self.post:
                continue
            wq = (1 + math.log(c)) * self.idf_tf[t]
            qn += wq * wq
            for i, tf in self.post[t].items():
                s[i] += wq * (1 + math.log(tf)) * self.idf_tf[t]
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
        if st + CHUNK_WORDS >= len(w):
            break
CH_DOC   = np.array([c[0] for c in CHUNKS])
CH_START = np.r_[0, np.flatnonzero(np.diff(CH_DOC)) + 1]

emb_status_detail = "belum diproses"

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
    f"[STARTUP] SEM_OK={SEM_OK} | EMBED_MODEL={EMBED_MODEL} "
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
    """Kembalikan array indeks dokumen berdasarkan z-score skor semantik (SEM_Z) dan SEM_TOP_K."""
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return np.array([], dtype=int)
    vals = sem_scores[valid]
    mean_sc = float(np.mean(vals))
    std_sc = float(np.std(vals))
    if std_sc > 1e-12:
        z_scores = (vals - mean_sc) / std_sc
        keep = valid[z_scores >= SEM_Z]
    else:
        keep = valid
    keep = keep[np.argsort(-sem_scores[keep], kind="stable")]
    if len(keep) == 0:
        ranked_valid = valid[np.argsort(-sem_scores[valid], kind="stable")]
        return ranked_valid[:5]
    if SEM_TOP_K:
        keep = keep[:SEM_TOP_K]
    return keep

def best_chunk(i, sims):
    s = CH_START[i]
    e = CH_START[i + 1] if i + 1 < len(CH_START) else len(sims)
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

def filter_mask(topik="Semua Topik", waktu="Semua Waktu", media="Semua Media"):
    m = np.ones(len(df), bool)
    if topik and not str(topik).startswith("Semua"):
        m &= (df["topic"] == topik).values
    if media and not str(media).startswith("Semua"):
        m &= (df["media"] == media).values
    if waktu and not str(waktu).startswith("Semua") and df["date"].notna().any():
        if str(waktu).startswith("Tahun"):
            m &= (df["date"].dt.year == int(str(waktu).split()[-1])).values
        else:
            days = {"7 hari terakhir": 7, "30 hari terakhir": 30, "1 tahun terakhir": 365}[waktu]
            m &= (df["date"] >= df["date"].max() - pd.Timedelta(days=days)).values
    return m

def _top_debug_rows(doc_indices, score_arr, limit=5):
    rows = []
    for i in doc_indices[:limit]:
        idx_int = int(i)
        r = df.iloc[idx_int]
        rows.append((str(r["doc_id"]), float(score_arr[idx_int]), str(r["title"])[:60]))
    return rows

def search(query, model="BM25", topik="Semua Topik", waktu="Semua Waktu", media="Semua Media", top_n=None, k1=1.5, b=0.75):
    t0 = time.perf_counter()
    if model in ("Semantic", "Hybrid") and not SEM_OK:
        model = "BM25"
    terms = preprocess(query)
    lex_terms = terms
    debug_enabled = os.environ.get("DEBUG_SEARCH", "").strip() == "1"
    mask, n = filter_mask(topik, waktu, media), len(df)
    lex, sem, sims = np.zeros(n), np.zeros(n), None
    sem_keep = np.array([], dtype=int)
    lex_keep = np.array([], dtype=int)

    if model == "Hybrid":
        if terms:
            max_df_count = LEX_MAX_DF * INDEX.N
            filtered_terms = [
                t for t in terms
                if (len(INDEX.post.get(t, {})) == 0) or (len(INDEX.post.get(t, {})) <= max_df_count)
            ]
            lex_terms = filtered_terms if filtered_terms else terms
        if lex_terms:
            lex = INDEX.bm25(lex_terms, k1, b)
        sem, sims = semantic_scores(query)
        sem_keep = filter_semantic_candidates(sem, mask)
        lex_valid = np.where(mask & (lex > 0))[0]
        lex_keep = lex_valid[np.argsort(-lex[lex_valid], kind="stable")] if len(lex_valid) > 0 else np.array([], dtype=int)
        sc = np.zeros(n)
        if len(lex_keep) > 0:
            sc[lex_keep] += HYBRID_W_LEX * (1.0 / (HYBRID_K + np.arange(1, len(lex_keep) + 1)))
        if len(sem_keep) > 0:
            sc[sem_keep] += HYBRID_W_SEM * (1.0 / (HYBRID_K + np.arange(1, len(sem_keep) + 1)))
        idx = np.where((sc > 0) & mask)[0]
        idx = idx[np.argsort(-sc[idx], kind="stable")]
    elif model == "Semantic":
        sem, sims = semantic_scores(query)
        sem_keep = filter_semantic_candidates(sem, mask)
        sc = np.zeros(n)
        sc[sem_keep] = sem[sem_keep]
        idx = sem_keep
    else:
        if terms:
            lex = INDEX.tfidf(terms) if model == "TF-IDF" else INDEX.bm25(terms, k1, b)
        sc = lex
        idx = np.where((sc > 0) & mask)[0]
        idx = idx[np.argsort(-sc[idx], kind="stable")]

    if top_n:
        idx = idx[:top_n]
    hits = [(int(i), float(sc[i]), best_chunk(int(i), sims) if sims is not None else None) for i in idx]
    if model == "Hybrid" and RERANKER and hits:
        hits = rerank_hits(query, hits, top_rerank=20)
        if top_n:
            hits = hits[:top_n]

    if debug_enabled:
        print(f"[DEBUG_SEARCH] model={model} | query={query!r}")
        print(f"  lexical_terms_before={terms}")
        print(f"  lexical_terms_after={lex_terms}")
        if model in ("BM25", "TF-IDF"):
            lex_rank = np.where(mask & (lex > 0))[0]
            lex_rank = lex_rank[np.argsort(-lex[lex_rank], kind="stable")]
            print(f"  top5_lexical={_top_debug_rows(lex_rank, lex, 5)}")
        elif model == "Semantic":
            print(f"  top5_semantic={_top_debug_rows(sem_keep, sem, 5)}")
        elif model == "Hybrid":
            print(f"  top5_lexical={_top_debug_rows(lex_keep, lex, 5)}")
            print(f"  top5_semantic={_top_debug_rows(sem_keep, sem, 5)}")
        final_rows = [
            (str(df.iloc[doc_i]["doc_id"]), float(final_sc), str(df.iloc[doc_i]["title"])[:60])
            for doc_i, final_sc, _ in hits[:5]
        ]
        print(f"  top5_final={final_rows}")

    return hits, terms, time.perf_counter() - t0

# Daftar Sistem Pencarian
SYSTEMS = {
    "TF-IDF": dict(model="TF-IDF"),
    "BM25": dict(model="BM25"),
}
if SEM_OK:
    SYSTEMS.update({
        "Semantic": dict(model="Semantic"),
        "Hybrid": dict(model="Hybrid"),
    })

def load_test_queries():
    if os.path.exists(FILE_QUERIES):
        qdf = pd.read_csv(FILE_QUERIES)
        return list(zip(qdf["query_id"].astype(str), qdf["query"].astype(str)))
    default_q = [
        "Jadwal Timnas U20 di kualifikasi piala asia",
        "Pemain diaspora yang baru dinaturalisasi",
        "Strategi pelatih Shin Tae-yong melawan Jepang",
        "Hasil pertandingan basket 3x3 putri SEA Games",
        "Upaya peningkatan kompetensi pelatih angkat besi",
        "Dukungan suporter di stadion GBK",
        "Kualitas tim sepak bola putri Garuda Pertiwi",
        "Persiapan atlet disabilitas menuju Paralimpiade",
        "Tanggapan Ketum PSSI mengenai prestasi E-Sports",
        "Daftar pemain yang dicoret dari skuad",
    ]
    return [(f"Q{i:02d}", q) for i, q in enumerate(default_q, 1)]

TEST_QUERIES = [q for _, q in load_test_queries()]

def buat_template_qrels():
    """Pooling kedalaman 20 dari semua sistem tanpa menimpa label lama di qrels.csv."""
    existing_labels = {}
    if os.path.exists(FILE_QRELS):
        old_df = pd.read_csv(FILE_QRELS)
        for _, r in old_df.iterrows():
            if pd.notna(r.get("relevan")) and str(r.get("relevan")).strip() != "":
                existing_labels[(str(r["query_id"]), str(r["doc_id"]))] = int(float(r["relevan"]))

    rows = []
    for qid, q in load_test_queries():
        pool = []
        seen = set()
        for cfg in SYSTEMS.values():
            for i, *_ in search(q, top_n=20, **cfg)[0]:
                if i not in seen:
                    seen.add(i)
                    pool.append(i)
        for i in pool:
            did = str(df.doc_id[i])
            lbl = existing_labels.get((qid, did), "")
            rows.append(dict(query_id=qid, query=q, doc_id=did, title=df.title[i], relevan=lbl))
    pd.DataFrame(rows).to_csv(FILE_QRELS_TEMPLATE, index=False)
    print(f"{FILE_QRELS_TEMPLATE} berhasil dibuat (pooling depth 20). Lengkapi kolom 'relevan' lalu simpan sebagai qrels.csv.")
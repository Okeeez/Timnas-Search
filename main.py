import os, time, math, base64, html, hashlib, re
import numpy as np, pandas as pd
import gradio as gr
from collections import defaultdict, Counter
from preprocessing import preprocess, stem_token  # Import dari file terpisah

# ===== KONFIGURASI =====
EMBED_MODEL  = "paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_WORDS, CHUNK_STRIDE = 70, 60
FILE_CORPUS = "Korpus_Bersih.xlsx"  # Gunakan korpus yang sudah dibersihkan
FILE_QRELS  = "qrels.csv"
DOSEN       = "Dr. Nurul Chamidah, S.Kom., M.Kom."
# Silakan isi NIM anggota tim di sini
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
print(f"Index siap: {INDEX.N} dokumen, {len(INDEX.post)} term unik.")

# Semantic Index
CHUNKS = []
for i, r in df.iterrows():
    w = r["text"].split() or [r["title"]]
    for st in range(0, len(w), CHUNK_STRIDE):
        seg = " ".join(w[st:st + CHUNK_WORDS])
        CHUNKS.append((i, seg, f'{r["title"]}. {seg}'))
        if st + CHUNK_WORDS >= len(w): break
CH_DOC   = np.array([c[0] for c in CHUNKS])
CH_START = np.r_[0, np.flatnonzero(np.diff(CH_DOC)) + 1]

try:
    from sentence_transformers import SentenceTransformer
    SEM_MODEL = SentenceTransformer(EMBED_MODEL)
    key = hashlib.md5((EMBED_MODEL + "|".join(c[2] for c in CHUNKS)).encode()).hexdigest()[:10]
    emb_file = f"emb_{key}.npy"
    
    if os.path.exists(emb_file):
        CH_EMB = np.load(emb_file)
    else:
        print("Mengunduh & memproses embedding semantic (sekitar 470MB)...")
        CH_EMB = SEM_MODEL.encode([c[2] for c in CHUNKS], batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        np.save(emb_file, CH_EMB)
    SEM_OK = True
    print(f"Semantic index siap: {len(CHUNKS)} potongan.")
except Exception as e:
    SEM_OK, SEM_MODEL, CH_EMB = False, None, None
    print("⚠️ Semantic search tidak aktif:", e)

def semantic_scores(query):
    q = SEM_MODEL.encode([query], normalize_embeddings=True)[0]
    sims = CH_EMB @ q
    return np.maximum.reduceat(sims, CH_START), sims

def best_chunk(i, sims):
    s = CH_START[i]; e = CH_START[i + 1] if i + 1 < len(CH_START) else len(sims)
    return CHUNKS[s + int(np.argmax(sims[s:e]))][1]

# Perbaikan Query Expansion (hanya memuat bobot makna yang relevan)
EXPANSION = {
    "timnas": ["garuda", "skuad"],
    "sty": ["shin tae-yong"],
    "erick": ["thohir", "ketum", "pssi"],
    "pelatih": ["coach"],
    "suporter": ["penonton", "fans"],
    "naturalisasi": ["diaspora"],
    "kualifikasi": ["putaran", "babak"]
}

def expand_query(q):
    ql = q.lower(); extra = []
    for k, v in EXPANSION.items():
        if re.search(rf"\b{re.escape(k)}\b", ql): extra += v
    return q + " " + " ".join(extra)

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

def search(query, model="BM25", expand=False, topik="Semua", waktu="Semua", media="Semua", top_n=None, k1=1.5, b=0.75, sem_min=0.30):
    t0 = time.perf_counter()
    if model in ("Semantic", "Hybrid") and not SEM_OK: model = "BM25"
    terms = preprocess(expand_query(query) if expand else query)
    mask, n = filter_mask(topik, waktu, media), len(df)
    lex, sem, sims = np.zeros(n), np.zeros(n), None
    
    if model != "Semantic" and terms: lex = INDEX.tfidf(terms) if model == "TF-IDF" else INDEX.bm25(terms, k1, b)
    if model in ("Semantic", "Hybrid"): sem, sims = semantic_scores(query)
    
    if model == "Semantic": sc = np.where(sem >= sem_min, sem, 0.0)
    elif model == "Hybrid":
        sc = np.zeros(n)
        for s_, ok in ((lex, lex > 0), (sem, sem >= sem_min)):
            ix = np.where(ok & mask)[0]; ix = ix[np.argsort(-s_[ix], kind="stable")]
            sc[ix] += 1.0 / (60 + np.arange(1, len(ix) + 1))
    else: sc = lex
    
    idx = np.where((sc > 0) & mask)[0]; idx = idx[np.argsort(-sc[idx], kind="stable")]
    if top_n: idx = idx[:top_n]
    hits = [(int(i), float(sc[i]), best_chunk(int(i), sims) if sims is not None else None) for i in idx]
    return hits, terms, time.perf_counter() - t0

# Query Pengujian Seimbang (Fakta, Parafrase, Semantic, Expansion)
SYSTEMS = {"TF-IDF (baseline)": dict(model="TF-IDF", expand=False),
           "BM25": dict(model="BM25", expand=False),
           "BM25 + Query Expansion": dict(model="BM25", expand=True)}
if SEM_OK: SYSTEMS.update({"Semantic": dict(model="Semantic"), "Hybrid (BM25 + Semantic)": dict(model="Hybrid")})

TEST_QUERIES = [
    "Jadwal Timnas U20 di kualifikasi piala asia",             # Fact / u20 regex test
    "Pemain diaspora yang baru dinaturalisasi",                # Expansion test
    "Strategi pelatih Shin Tae-yong melawan Jepang",           # Entity
    "Hasil pertandingan basket 3x3 putri SEA Games",           # KATA_PENTING test (3x3, sea)
    "Upaya peningkatan kompetensi pelatih angkat besi",        # Semantic paraphrase
    "Dukungan suporter di stadion GBK",                        # Expansion
    "Kualitas tim sepak bola putri Garuda Pertiwi",            # Fact
    "Persiapan atlet disabilitas menuju Paralimpiade",         # Semantic
    "Tanggapan Ketum PSSI mengenai prestasi E-Sports",         # Expansion & Protected words
    "Daftar pemain yang dicoret dari skuad"                    # Paraphrase (pemecatan/dicoret)
]

def buat_template_qrels():
    rows = []
    for qi, q in enumerate(TEST_QUERIES, 1):
        pool = {}
        for cfg in SYSTEMS.values():
            for i, *_ in search(q, top_n=10, **cfg)[0]: pool[i] = 1
        rows += [dict(query_id=f"Q{qi}", query=q, doc_id=df.doc_id[i], title=df.title[i], relevan="")]
    pd.DataFrame(rows).to_csv("qrels_template.csv", index=False)
    print("qrels_template.csv berhasil dibuat. Isi kolom 'relevan' dengan 1 atau 0, lalu simpan sebagai qrels.csv.")

# UI Gradio (Kode UI statis tidak diubah, dilewati untuk efisiensi panduan, 
# Anda bisa menyalin blok UI (Gradio CSS & Blocks) dari notebook awal tanpa perubahan 
# kecuali fungsi background dan text layout jika diperlukan).

if __name__ == "__main__":
    if not os.path.exists("qrels_template.csv"): buat_template_qrels()
    
    # ------------------ BLOK UI GRADIO ------------------
    BG_IMAGE = "suporter.jpg"

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
        text = text[:350] + "..." if len(text) > 350 else text
        if chunk:
            c_idx = text.lower().find(chunk.lower()[:30])
            if c_idx != -1: text = ("..." if c_idx > 50 else "") + text[max(0, c_idx-50):c_idx+250] + "..."
        if not terms: return html.escape(text)
        
        esc = html.escape(text)
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

    def do_search(q, mod_label, exp_checkbox, top, wak, med):
        # Jika query kosong, batalkan pencarian
        if not q.strip(): 
            return "<div>Masukkan kata kunci.</div>", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), []
        
        # 1. BONGKAR DICTIONARY SYSTEMS
        # Ambil konfigurasi asli berdasarkan label yang dipilih di dropdown UI
        config = SYSTEMS.get(mod_label, {"model": "BM25", "expand": False})
        actual_model = config["model"]
        
        # 2. GABUNGKAN LOGIKA EKSPANSI
        # Aktifkan ekspansi jika dropdown meminta ekspansi ATAU checkbox dicentang
        use_expansion = config["expand"] or exp_checkbox
        
        # 3. PANGGIL FUNGSI SEARCH DENGAN PARAMETER YANG BENAR
        h, t, dt = search(q, model=actual_model, expand=use_expansion, topik=top, waktu=wak, media=med)
        
        # Tampilkan hasil
        out, p, n, l = hasil_html(h, t, dt, q, 1)
        return out, p, n, l, {"hits": h, "terms": t, "dt": dt, "q": q, "page": 1}

    def change_page(state, dir_):
        if not state: return "<div>Error</div>", gr.update(), gr.update(), gr.update(), state
        state["page"] += dir_
        out, p, n, l = hasil_html(state["hits"], state["terms"], state["dt"], state["q"], state["page"])
        return out, p, n, l, state

    with gr.Blocks(title="Timnas Search", theme=gr.themes.Base()) as demo:
        state = gr.State({})
        
        # Header (Logo & Search Bar sejajar ala Mockup 2)
        with gr.Row(elem_classes="top-header"):
            gr.HTML("<div class='logo-img'>⚽ Timnas <span>Search</span></div>", scale=1)
            with gr.Column(scale=4, elem_classes="search-box-container"):
                q_in = gr.Textbox(placeholder="Ketik pencarian berita timnas...", show_label=False, container=False, lines=1)
        
        # Pill Filters (Di bawah search bar)
        with gr.Row(elem_classes="filter-row"):
            top_in = gr.Dropdown(["Semua"] + sorted([str(x) for x in df['topic'].unique() if str(x).strip()]), value="Semua", label="Topik", container=False, elem_classes="pill-dropdown")
            wak_in = gr.Dropdown(["Semua", "7 hari terakhir", "30 hari terakhir", "1 tahun terakhir"], value="Semua", label="Waktu", container=False, elem_classes="pill-dropdown")
            med_in = gr.Dropdown(["Semua"] + sorted([str(x) for x in df['media'].unique() if str(x).strip()]), value="Semua", label="Media", container=False, elem_classes="pill-dropdown")
            mod_in = gr.Dropdown(list(SYSTEMS.keys()), value="BM25", label="Model (Korpus N)", container=False, elem_classes="pill-dropdown")
            exp_in = gr.Checkbox(label="Expansion", value=False, elem_classes="pill-dropdown")
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
        btn.click(do_search, [q_in, mod_in, exp_in, top_in, wak_in, med_in], [out_html, prev_b, next_b, page_l, state])
        q_in.submit(do_search, [q_in, mod_in, exp_in, top_in, wak_in, med_in], [out_html, prev_b, next_b, page_l, state])
        prev_b.click(lambda s: change_page(s, -1), state, [out_html, prev_b, next_b, page_l, state])
        next_b.click(lambda s: change_page(s, 1), state, [out_html, prev_b, next_b, page_l, state])

    # Eksekusi dengan inject CSS dan background image script
    demo.launch(css=bg_rule() + CSS_STATIC)
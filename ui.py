# ui.py
import os, math, base64, html, re
import pandas as pd
import gradio as gr
from engine import df, search, SYSTEMS, SEM_OK, TIM, DOSEN, PER_PAGE, BASE_DIR, stem_token


def _build_card_bg_var() -> str:
    for fname in ("stadion.jpg", "suporter.jpg"):
        fpath = os.path.join(BASE_DIR, fname)
        if os.path.exists(fpath):
            with open(fpath, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"url('data:image/jpeg;base64,{b64}')"
    return "linear-gradient(135deg, #2d0a10 0%, #1a1a24 55%, #0f172a 100%)"


def build_css() -> str:
    with open(os.path.join(BASE_DIR, "static", "style.css"), encoding="utf-8") as f:
        css = f.read()
    return f":root {{ --card-img: {_build_card_bg_var()}; }}\n" + css


def _score_label(model_name):
    mapping = {
        "BM25": "Skor BM25",
        "TF-IDF": "Skor TF-IDF",
        "Semantic": "Skor cosine",
        "Hybrid": "Skor RRF",
    }
    return mapping.get(model_name, "Skor")


def make_snippet(text, terms, chunk):
    if not text:
        return ""
    if chunk:
        src_text = chunk
    else:
        src_text = text[:350] + "..." if len(text) > 350 else text
    if not terms:
        return html.escape(src_text)

    term_set = set(terms)
    parts = []
    last = 0
    for m in re.finditer(r"\w+", src_text):
        if m.start() > last:
            parts.append(html.escape(src_text[last:m.start()]))
        word = m.group(0)
        esc_word = html.escape(word)
        if stem_token(word.lower()) in term_set:
            parts.append(f"<mark>{esc_word}</mark>")
        else:
            parts.append(esc_word)
        last = m.end()
    if last < len(src_text):
        parts.append(html.escape(src_text[last:]))
    return "".join(parts)


def card_html(row, score, snippet, model_name="BM25"):
    dt = row['date'].strftime('%d/%m/%Y') if pd.notnull(row.get('date')) else 'Tanpa Tanggal'
    score_lbl = _score_label(model_name)
    topic_txt = html.escape(str(row['topic']).strip() or "BERITA")
    title_txt = html.escape(str(row['title']))
    media_txt = html.escape(str(row['media']))
    url_txt = html.escape(str(row['url']), quote=True)
    chip_txt = f"{media_txt} • {score_lbl}: {score:.2f}"

    cal_svg = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>'
        '<line x1="16" y1="2" x2="16" y2="6"></line>'
        '<line x1="8" y1="2" x2="8" y2="6"></line>'
        '<line x1="3" y1="10" x2="21" y2="10"></line>'
        '</svg>'
    )

    return f"""
    <a class="news-card" href="{url_txt}" target="_blank" rel="noopener">
        <div class="card-bg"></div>
        <div class="card-chip">{chip_txt}</div>
        <div class="card-content">
            <span class="card-topic-badge">{topic_txt}</span>
            <h3 class="card-title">{title_txt}</h3>
            <div class="card-divider"></div>
            <div class="card-date-row">{cal_svg}<span>{html.escape(dt)}</span></div>
        </div>
        <div class="card-snippet-overlay">
            <div class="card-snippet-text">{snippet}</div>
        </div>
    </a>
    """


def hasil_html(hits, terms, dt, query, shown, model_name="BM25", cached_cards=None):
    if not hits:
        return (
            f"<div class='empty-state'>Tidak ada hasil untuk <b>'{html.escape(query)}'</b>.</div>",
            gr.update(visible=False),
        )

    total = len(hits)
    shown = min(shown, total)

    cards = list(cached_cards) if cached_cards else []
    if len(cards) < shown:
        for idx, score, chunk in hits[len(cards):shown]:
            row = df.iloc[idx]
            snip = make_snippet(row['text'], terms, chunk)
            cards.append(card_html(row, score, snip, model_name=model_name))
    if cached_cards is not None and len(cached_cards) < len(cards):
        cached_cards.extend(cards[len(cached_cards):])

    end_note = f"<div class='end-note'>Semua {total} hasil sudah ditampilkan</div>" if shown >= total else ""
    out = (
        f"<div class='stats-bar'>{total} hasil ({dt:.2f} detik) • menampilkan {shown} dari {total}</div>"
        f"<div class='card-grid'>{''.join(cards[:shown])}</div>"
        f"{end_note}"
    )

    return out, gr.update(visible=shown < total)


def do_search(q, mod_label, top, wak, med):
    if not q.strip():
        return "<div class='empty-state'>Masukkan kata kunci.</div>", gr.update(visible=False), {}

    config = SYSTEMS.get(mod_label, {"model": "BM25"})
    actual_model = config["model"]

    h, t, dt = search(q, model=actual_model, topik=top, waktu=wak, media=med)
    shown = min(PER_PAGE, len(h))
    cards = []

    out, more_upd = hasil_html(h, t, dt, q, shown, model_name=actual_model, cached_cards=cards)
    new_state = {
        "hits": h,
        "terms": t,
        "dt": dt,
        "q": q,
        "model": actual_model,
        "shown": shown,
        "cards": cards,
    }
    return out, more_upd, new_state


def load_more(state):
    if not state or not state.get("hits"):
        return gr.update(), gr.update(visible=False), state

    total = len(state["hits"])
    if state.get("shown", 0) >= total:
        return gr.update(), gr.update(visible=False), state

    state["shown"] = min(state["shown"] + PER_PAGE, total)
    cards = state.setdefault("cards", [])
    out, more_upd = hasil_html(
        state["hits"],
        state["terms"],
        state["dt"],
        state["q"],
        state["shown"],
        model_name=state.get("model", "BM25"),
        cached_cards=cards,
    )
    return out, more_upd, state


AUTO_LOAD_HEAD = """
<script>
(function () {
  let busy = false;
  setInterval(function () {
    if (busy) return;
    const wrap = document.querySelector('#load-more-btn');
    if (!wrap) return;
    const btn = wrap.tagName === 'BUTTON' ? wrap : wrap.querySelector('button');
    if (!btn || btn.disabled) return;
    const r = btn.getBoundingClientRect();
    const visible = r.width > 0 && r.height > 0;
    if (visible && r.top < window.innerHeight + 400) {
      busy = true;
      btn.click();
      setTimeout(function () { busy = false; }, 900);
    }
  }, 300);
})();
</script>
"""

light_theme = gr.themes.Base(primary_hue="red", neutral_hue="slate")


def build_demo() -> gr.Blocks:
    default_model_val = "Hybrid" if SEM_OK else "BM25"

    with gr.Blocks(title="Timnas Search") as demo:
        state = gr.State({})

        # Garis aksen merah tipis di paling atas
        gr.HTML("<div class='top-accent-bar'></div>")

        # Header: Logo Teks Kiri & Search Pill Kanan
        with gr.Row(elem_classes="top-header"):
            gr.HTML("<div class='brand-logo'>TIMNAS <span>SEARCH</span></div>")
            with gr.Column(elem_classes="search-box-container"):
                q_in = gr.Textbox(
                    placeholder="Ketik pencarian berita timnas...",
                    show_label=False,
                    container=False,
                    lines=1,
                )

        if not SEM_OK:
            gr.HTML("<div class='sem-notice'>Semantic search tidak aktif, memakai BM25</div>")

        # Tabs Layout
        with gr.Tabs(elem_classes="tabs"):
            with gr.TabItem("Search Engine"):
                # Baris Filter di dalam Tab Search Engine
                with gr.Row(elem_classes="filter-row"):
                    top_in = gr.Dropdown(
                        ["Semua Topik"] + sorted([str(x) for x in df['topic'].unique() if str(x).strip()]),
                        value="Semua Topik",
                        label="Topik",
                        container=False,
                        elem_classes="pill-dropdown",
                    )
                    wak_in = gr.Dropdown(
                        ["Semua Waktu", "7 hari terakhir", "30 hari terakhir", "1 tahun terakhir"],
                        value="Semua Waktu",
                        label="Waktu",
                        container=False,
                        elem_classes="pill-dropdown",
                    )
                    med_in = gr.Dropdown(
                        ["Semua Media"] + sorted([str(x) for x in df['media'].unique() if str(x).strip()]),
                        value="Semua Media",
                        label="Media",
                        container=False,
                        elem_classes="pill-dropdown",
                    )
                    mod_in = gr.Dropdown(
                        list(SYSTEMS.keys()),
                        value=default_model_val,
                        label="Model (Korpus N)",
                        container=False,
                        elem_classes="pill-dropdown",
                    )
                    btn = gr.Button("Search", variant="primary", elem_classes="search-btn-pill")

                out_html = gr.HTML("<div class='empty-state'>Masukkan kata kunci pencarian berita timnas di atas.</div>")
                more_b = gr.Button("Muat lebih banyak", visible=False, elem_id="load-more-btn", elem_classes="load-more-pill")

            with gr.TabItem("Detail Korpus"):
                gr.HTML(f"<h2 class='section-heading'>TOTAL KORPUS: {len(df)} BERITA</h2>")
                gr.Dataframe(
                    df[['date', 'title', 'media', 'topic']].head(50),
                    elem_classes="corpus-table-wrap",
                )

            with gr.TabItem("Tentang Tim"):
                gr.HTML("<h2 class='section-heading'>ANGGOTA TIM PENGEMBANG</h2>")
                team_cards = "".join(
                    f"<div class='member-card'><div class='member-name'>{html.escape(n)}</div><div class='member-nim'>NIM: {html.escape(i)}</div></div>"
                    for n, i in TIM
                )
                gr.HTML(
                    f"<div class='team-grid'>{team_cards}</div>"
                    f"<div class='course-info-box'><b>Dosen Pengampu:</b> {html.escape(DOSEN)}<br>"
                    f"<b>Mata Kuliah:</b> Temu Kembali Informasi</div>"
                )

        # Footer Fixed di Bawah
        nama_tim_str = ", ".join(html.escape(n) for n, _ in TIM)
        gr.HTML(f"""
        <div class="fixed-footer">
            <div>Kredit tim: {nama_tim_str}</div>
            <div>Dosen/Mata Kuliah: {html.escape(DOSEN)} — Temu Kembali Informasi</div>
        </div>
        """)

        # Event Listeners
        btn.click(
            do_search,
            [q_in, mod_in, top_in, wak_in, med_in],
            [out_html, more_b, state],
        ).then(fn=None, js="() => window.scrollTo({top: 0, behavior: 'smooth'})")

        q_in.submit(
            do_search,
            [q_in, mod_in, top_in, wak_in, med_in],
            [out_html, more_b, state],
        ).then(fn=None, js="() => window.scrollTo({top: 0, behavior: 'smooth'})")

        more_b.click(load_more, state, [out_html, more_b, state])

    return demo


def launch():
    demo = build_demo()
    demo.launch(
        theme=light_theme,
        css=build_css(),
        head=AUTO_LOAD_HEAD,
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=True

    )

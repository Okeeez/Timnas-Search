# main.py
import os, re
import gradio as gr


def _gradio_major_version(ver_str: str) -> int:
    m = re.match(r"^\s*(\d+)", str(ver_str))
    return int(m.group(1)) if m else 0


GRADIO_VER = getattr(gr, "__version__", "0.0.0")
if _gradio_major_version(GRADIO_VER) < 6:
    print(f"ERROR: Aplikasi ini membutuhkan gradio>=6.0, versi terpasang: {GRADIO_VER}")
    raise SystemExit(1)

from engine import (  # noqa: E402
    FILE_QRELS_TEMPLATE,
    buat_template_qrels,
    search,
    SYSTEMS,
    df,
    load_test_queries,
    TEST_QUERIES,
)
import ui  # noqa: E402


if __name__ == "__main__":
    if not os.path.exists(FILE_QRELS_TEMPLATE):
        buat_template_qrels()
    ui.launch()

demo.launch(
    theme=gr.themes.Base(),
    css=bg_rule() + CSS_STATIC,
    server_name="0.0.0.0",
    server_port=int(os.environ.get("PORT", 7860)),
    share=True,  # <-- Wajib agar muncul link publik saat di-run di GitHub
)

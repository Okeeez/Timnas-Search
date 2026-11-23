# evaluasi.py - menghitung Precision@k, Recall@k, F1@10, dan MAP dari qrels.csv
import numpy as np, pandas as pd
import main as m   # aman diimpor: UI hanya jalan di bawah if __name__ == "__main__"

KS = (5, 10, 20)

qr = pd.read_csv(m.FILE_QRELS)
qr["relevan"] = pd.to_numeric(qr["relevan"], errors="coerce").fillna(0).astype(int)

rows = []
for label, cfg in m.SYSTEMS.items():
    for qid, g in qr.groupby("query_id", sort=False):
        q = g["query"].iloc[0]
        rel = set(g.loc[g["relevan"] == 1, "doc_id"])
        R = len(rel)
        if R == 0:
            continue                      # query tanpa dokumen relevan tidak bisa dihitung recall-nya
        hits, _, _ = m.search(q, **cfg)   # daftar hasil penuh (bukan hanya top-10)
        ranked = [m.df.doc_id[i] for i, _, _ in hits]
        flags = np.array([d in rel for d in ranked], dtype=float)

        row = {"sistem": label, "query_id": qid, "R": R, "jumlah_hasil": len(ranked)}
        for k in KS:
            tp = flags[:k].sum()
            row[f"P@{k}"] = tp / k
            row[f"R@{k}"] = tp / R
        p10, r10 = row["P@10"], row["R@10"]
        row["F1@10"] = 2 * p10 * r10 / (p10 + r10) if (p10 + r10) else 0.0
        row["Recall_penuh"] = flags.sum() / R           # recall atas seluruh daftar hasil
        prec_at_rank = np.cumsum(flags) / (np.arange(len(flags)) + 1)
        row["AP"] = float((prec_at_rank * flags).sum() / R)
        rows.append(row)

detail = pd.DataFrame(rows)
detail.to_csv("evaluasi_per_query.csv", index=False)

ringkas = detail.groupby("sistem", sort=False)[
    [f"P@{k}" for k in KS] + [f"R@{k}" for k in KS] + ["F1@10", "Recall_penuh", "AP"]
].mean().rename(columns={"AP": "MAP"}).round(3)
ringkas.to_csv("evaluasi_ringkasan.csv")
print(ringkas.to_string())
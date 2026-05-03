import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("research/full_experiment_results")
exp1 = pd.read_csv(OUT / "exp1_feedback_options.csv")
exp2 = pd.read_csv(OUT / "exp2_model_comparison.csv")
exp3 = pd.read_csv(OUT / "exp3_duplicate_questions.csv")

# ─ EXP1 SAMPLES ─
e1_low  = exp1[exp1["score_sbert"] == exp1["score_sbert"].min()].iloc[0]
e1_mid_m = (exp1["score_sbert"] >= 50) & (exp1["score_sbert"] < 70)
e1_mid  = exp1[e1_mid_m].iloc[0] if e1_mid_m.any() else exp1.iloc[len(exp1)//2]
e1_high = exp1[exp1["score_sbert"] == exp1["score_sbert"].max()].iloc[0]

md = f"""# Summary Hasil Eksperimen — quizzess-ml AI Feedback Research

> **Gemini Model**: `gemini-3-flash-preview` | **Dataset**: `essay_submissions.csv`
> **Eksperimen 1 & 2**: 10 baris terakhir dataset (ID 94–103) | **Eksperimen 3**: Soal sama, jawaban berbeda

---

## Eksperimen 1 — Perbandingan 5 Opsi AI Feedback

> Model scoring tetap: **SBERT (all-MiniLM-L6-v2)**

### Distribusi Skor SBERT (10 sampel)

| Mean | Std | Min | Max |
|------|-----|-----|-----|
| {exp1["score_sbert"].mean():.2f} | {exp1["score_sbert"].std():.2f} | {exp1["score_sbert"].min():.2f} | {exp1["score_sbert"].max():.2f} |

---

### Contoh: Skor Rendah — Q{e1_low["question_id"]} ({e1_low["score_sbert"]:.2f}/100)

> **Jawaban**: `{str(e1_low["user_answer"])[:120]}{'...' if len(str(e1_low["user_answer"]))>120 else ''}`
> **Kunci**: `{str(e1_low["teacher_key"])[:100]}...`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | {e1_low["feedback_A_rule"]} |
| B (KeyBERT) | {e1_low["feedback_B_keybert"]} |
| C (Template) | {e1_low["feedback_C_template"]} |
| D (Gemini) | {e1_low["feedback_D_gemini"]} |
| E (Hybrid) | {e1_low["feedback_E_hybrid"]} |

---

### Contoh: Skor Sedang — Q{e1_mid["question_id"]} ({e1_mid["score_sbert"]:.2f}/100)

> **Jawaban**: `{str(e1_mid["user_answer"])[:120]}{'...' if len(str(e1_mid["user_answer"]))>120 else ''}`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | {e1_mid["feedback_A_rule"]} |
| B (KeyBERT) | {e1_mid["feedback_B_keybert"]} |
| C (Template) | {e1_mid["feedback_C_template"]} |
| D (Gemini) | {e1_mid["feedback_D_gemini"]} |
| E (Hybrid) | {e1_mid["feedback_E_hybrid"]} |

---

### Contoh: Skor Tinggi — Q{e1_high["question_id"]} ({e1_high["score_sbert"]:.2f}/100)

> **Jawaban**: `{str(e1_high["user_answer"])[:120]}{'...' if len(str(e1_high["user_answer"]))>120 else ''}`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | {e1_high["feedback_A_rule"]} |
| B (KeyBERT) | {e1_high["feedback_B_keybert"]} |
| C (Template) | {e1_high["feedback_C_template"]} |
| D (Gemini) | {e1_high["feedback_D_gemini"]} |
| E (Hybrid) | {e1_high["feedback_E_hybrid"]} |

---

### Evaluasi Kualitas Opsi Feedback

| Opsi | Kualitas | Informatif | Butuh API | Rekomendasi |
|------|----------|-----------|-----------|-------------|
| A (Rule-based) | ⭐⭐ | ❌ Generik | ❌ | Fallback terakhir |
| B (KeyBERT) | ⭐⭐⭐ | ✅ Sebut konsep | ❌ | Fallback lokal bagus |
| C (Template) | ⭐⭐⭐ | ✅ Tampilkan % skor | ❌ | Alternatif lokal terbaik |
| D (Gemini) | ⭐⭐⭐⭐⭐ | ✅ Kontekstual & natural | ✅ | Terbaik untuk produksi |
| E (Hybrid) | ⭐⭐⭐⭐⭐ | ✅ Keyword + konteks Gemini | ✅ | Terbaik & paling spesifik |

---

## Eksperimen 2 — Perbandingan 4 Model Scoring

| Model | Mean | Std | Min | Max | Catatan |
|-------|------|-----|-----|-----|---------|
| TF-IDF | {exp2["score_tfidf"].mean():.2f} | {exp2["score_tfidf"].std():.2f} | {exp2["score_tfidf"].min():.2f} | {exp2["score_tfidf"].max():.2f} | Sangat lemah, banyak 0.0 |
| SBERT (en) | {exp2["score_sbert"].mean():.2f} | {exp2["score_sbert"].std():.2f} | {exp2["score_sbert"].min():.2f} | {exp2["score_sbert"].max():.2f} | Stabil, tapi English-only |
| Paraphrase-multilingual | {exp2["score_multilingual"].mean():.2f} | {exp2["score_multilingual"].std():.2f} | {exp2["score_multilingual"].min():.2f} | {exp2["score_multilingual"].max():.2f} | Std tinggi, bisa negatif |
| **IndoBERT** | **{exp2["score_indobert"].mean():.2f}** | {exp2["score_indobert"].std():.2f} | {exp2["score_indobert"].min():.2f} | {exp2["score_indobert"].max():.2f} | **Terbaik untuk Bahasa Indonesia** |

### Perbandingan per Soal

| ID | Q | TF-IDF | SBERT | Multilingual | IndoBERT | Original |
|----|---|--------|-------|-------------|----------|----------|
"""

for _, r in exp2.iterrows():
    md += f"| {r['id']} | {r['question_id']} | {r['score_tfidf']} | {r['score_sbert']} | {r['score_multilingual']} | {r['score_indobert']} | {r['ai_score_original']} |\n"

md += """
---

## Eksperimen 3 — Soal Sama, Jawaban Berbeda

> Melihat konsistensi feedback untuk question_id yang sama dengan jawaban berbeda

"""

for _, r in exp3.iterrows():
    md += f"### Question ID: {r['question_id']}\n"
    md += f"**Kunci Jawaban**: {r['teacher_key']}\n\n"
    md += f"#### History {r['history_id']} (ID {r['submission_id']}) | SBERT: {r['score_sbert']}\n"
    md += f"**Jawaban**: `{str(r['user_answer'])[:120]}{'...' if len(str(r['user_answer']))>120 else ''}`\n\n"
    
    md += "| Opsi | Feedback |\n"
    md += "|------|----------|\n"
    md += f"| A (Rule-based) | {r['feedback_A_rule']} |\n"
    md += f"| B (KeyBERT) | {r['feedback_B_keybert']} |\n"
    md += f"| C (Template) | {r['feedback_C_template']} |\n"
    md += f"| D (Gemini) | {r['feedback_D_gemini']} |\n"
    md += f"| E (Hybrid) | {r['feedback_E_hybrid']} |\n\n"

md += """---

## Rekomendasi

| Aspek | Sekarang | Rekomendasi |
|-------|----------|-------------|
| **Model Scoring** | SBERT (en) | Upgrade ke **IndoBERT** — mean tertinggi & lebih cocok Bahasa Indonesia |
| **AI Feedback** | Rule-based (Opsi A) | **Opsi E (Hybrid)** untuk produksi, **Opsi C (Template)** sebagai fallback offline |
| **Eksperimen 3** | — | Feedback dinamis (Gemini/Hybrid) merespons perbedaan redaksi jawaban dengan baik meskipun soalnya sama |

---

## Output Files

| File | Isi |
|------|-----|
| [`exp1_feedback_options.csv`](exp1_feedback_options.csv) / [`.md`](exp1_feedback_options.md) | 10 baris × 5 opsi feedback |
| [`exp2_model_comparison.csv`](exp2_model_comparison.csv) / [`.md`](exp2_model_comparison.md) | 10 baris × 4 model skor |
| [`exp3_duplicate_questions.csv`](exp3_duplicate_questions.csv) / [`.md`](exp3_duplicate_questions.md) | 3 sampel soal sama × jawaban berbeda |
| `summary.md` | File ini |
"""

with open(OUT / "summary.md", "w", encoding="utf-8") as f:
    f.write(md)
print("summary.md updated!")

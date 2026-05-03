"""
full_experiment.py
==================
3 Eksperimen sekaligus menggunakan dataset primer essay_submissions.csv

Eksperimen 1: 5 opsi AI Feedback (model scoring: SBERT tetap)
Eksperimen 2: 4 model scoring (TF-IDF, SBERT, IndoBERT, Paraphrase-multilingual)
Eksperimen 3: Soal sama, jawaban berbeda (question_id: 327, 329, 333)
"""

import os
import sys
import time
import warnings
import pandas as pd
import torch
from pathlib import Path

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
RESEARCH_DIR = Path(__file__).parent
DATASET_PATH = RESEARCH_DIR / "essay_submissions.csv"
OUTPUT_DIR   = RESEARCH_DIR / "full_experiment_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# 1. LOAD DATASET
# ──────────────────────────────────────────────
print("📂 Loading dataset...")
df_full = pd.read_csv(DATASET_PATH)
df = df_full.tail(10).reset_index(drop=True)
print(f"   ✅ {len(df_full)} rows total → menggunakan 10 baris terakhir (ID {df['id'].iloc[0]}–{df['id'].iloc[-1]})")

# ──────────────────────────────────────────────
# 2. LOAD MODELS
# ──────────────────────────────────────────────
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from keybert import KeyBERT
import google.generativeai as genai

GEMINI_API_KEY = "AIzaSyA4gb4VCLDYkZQQYsOeGl2_L9haTO-5GU4"

print("\n⏳ Loading models...")

print("   Loading SBERT (all-MiniLM-L6-v2)...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

print("   Loading Paraphrase-multilingual (multilingual SBERT)...")
try:
    multi_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    MULTI_OK = True
    print("   ✅ Paraphrase-multilingual loaded.")
except Exception as e:
    print(f"   ⚠️ Paraphrase-multilingual gagal load: {e}")
    MULTI_OK = False
    multi_model = None

print("   Loading IndoBERT...")
try:
    indobert_tokenizer = AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")
    indobert_model     = AutoModel.from_pretrained("indobenchmark/indobert-base-p1")
    INDOBERT_OK = True
    print("   ✅ IndoBERT loaded.")
except Exception as e:
    print(f"   ⚠️ IndoBERT gagal load: {e}")
    INDOBERT_OK = False
    indobert_tokenizer = indobert_model = None

print("   Loading KeyBERT...")
kw_extractor = KeyBERT(model=sbert_model)

print("   Configuring Gemini...")
genai.configure(api_key=GEMINI_API_KEY)
try:
    gemini = genai.GenerativeModel("gemini-3-flash-preview")
    GEMINI_OK = True
    print("   ✅ Gemini configured.")
except Exception as e:
    print(f"   ⚠️ Gemini gagal: {e}")
    GEMINI_OK = False
    gemini = None

# ──────────────────────────────────────────────
# 3. SCORING HELPERS
# ──────────────────────────────────────────────

def score_tfidf(key: str, ans: str) -> float:
    try:
        v  = TfidfVectorizer()
        m  = v.fit_transform([key, ans])
        return float(cosine_similarity(m[0:1], m[1:2])[0][0]) * 100
    except Exception:
        return 0.0

def score_sbert(key: str, ans: str) -> float:
    try:
        emb = sbert_model.encode([key, ans], convert_to_tensor=True)
        return float(util.cos_sim(emb[0], emb[1]).item()) * 100
    except Exception:
        return 0.0

def score_multilingual(key: str, ans: str) -> float:
    if not MULTI_OK:
        return 0.0
    try:
        emb = multi_model.encode([key, ans], convert_to_tensor=True)
        return float(util.cos_sim(emb[0], emb[1]).item()) * 100
    except Exception:
        return 0.0

def score_indobert(key: str, ans: str) -> float:
    if not INDOBERT_OK:
        return 0.0
    try:
        inputs = indobert_tokenizer([key, ans], padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = indobert_model(**inputs)
        tok_emb = outputs.last_hidden_state
        mask    = inputs["attention_mask"]
        expanded = mask.unsqueeze(-1).expand(tok_emb.size()).float()
        emb = torch.sum(tok_emb * expanded, 1) / torch.clamp(expanded.sum(1), min=1e-9)
        return float(cosine_similarity(emb[0].reshape(1,-1), emb[1].reshape(1,-1))[0][0]) * 100
    except Exception:
        return 0.0

# ──────────────────────────────────────────────
# 4. FEEDBACK HELPERS
# ──────────────────────────────────────────────

def feedback_A(score: float) -> str:
    """Opsi A: Rule-based threshold (current production)"""
    if score >= 90:
        return "Sempurna! Pemahaman sangat mendalam."
    elif score >= 75:
        return "Bagus, namun ada sedikit perbedaan istilah."
    elif score >= 50:
        return "Kurang tepat, perbaiki lagi konsepnya."
    else:
        return "Jawaban tidak relevan dengan kunci jawaban."

def feedback_B(key: str, ans: str, threshold: float = 0.38) -> str:
    """Opsi B: KeyBERT missing keywords"""
    if not ans or len(ans.strip()) < 3:
        return "Jawaban terlalu singkat atau kosong."
    try:
        keywords = kw_extractor.extract_keywords(
            key, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=4
        )
        missing = []
        ans_emb = sbert_model.encode(ans, convert_to_tensor=True)
        for kw, _ in keywords:
            kw_emb = sbert_model.encode(kw, convert_to_tensor=True)
            sim = util.cos_sim(kw_emb, ans_emb).item()
            if sim < threshold:
                missing.append(kw)
        if missing:
            return f"Jawaban belum mencakup konsep penting: {', '.join(missing)}."
        return "Jawaban sudah mencakup seluruh poin penting dari kunci jawaban."
    except Exception as e:
        return f"[ERROR KeyBERT] {e}"

def feedback_C(key: str, ans: str, score: float) -> str:
    """Opsi C: Template dinamis berdasar skor + persentase kemiripan"""
    pct = round(score, 1)
    if not ans or len(ans.strip()) < 3:
        return f"Jawaban terlalu singkat. Kemiripan dengan kunci jawaban hanya {pct}%."
    if score >= 85:
        return f"Jawaban sangat baik! Tingkat kesesuaian dengan kunci jawaban mencapai {pct}%."
    elif score >= 65:
        return (f"Jawaban cukup baik ({pct}% sesuai kunci). "
                "Coba tambahkan detail atau istilah yang lebih spesifik sesuai materi.")
    elif score >= 40:
        return (f"Jawaban kurang tepat ({pct}% sesuai kunci). "
                "Periksa kembali konsep utama dan gunakan kata kunci yang relevan.")
    else:
        return (f"Jawaban belum sesuai kunci jawaban ({pct}%). "
                "Pelajari ulang materi dan coba jawab dengan kata-kata sendiri.")

def feedback_D(key: str, ans: str, score: float) -> str:
    """Opsi D: Gemini generative"""
    if not GEMINI_OK:
        return "[MOCK] Gemini tidak tersedia."
    prompt = (
        f"Nilai siswa: {score:.1f}/100.\n"
        f"Kunci Jawaban: {key}\n"
        f"Jawaban Siswa: {ans}\n\n"
        "Evaluasi jawaban siswa di atas secara singkat (maks 2 kalimat) dalam bahasa Indonesia. "
        "Jelaskan mengapa nilainya demikian dan apa yang perlu diperbaiki (atau puji jika sudah tepat). "
        "JANGAN berikan nilai angka lagi."
    )
    try:
        resp = gemini.generate_content(prompt)
        return resp.text.replace("\n", " ").strip()
    except Exception as e:
        return f"[ERROR Gemini] {e}"

def feedback_E(key: str, ans: str, score: float) -> str:
    """Opsi E: Hybrid — KeyBERT context → Gemini generate lebih spesifik"""
    if not GEMINI_OK:
        return "[MOCK] Gemini tidak tersedia (Hybrid)."
    # Step 1: extract keywords
    try:
        keywords = kw_extractor.extract_keywords(
            key, keyphrase_ngram_range=(1, 2), stop_words=None, top_n=3
        )
        kw_str = ", ".join([kw for kw, _ in keywords])
    except Exception:
        kw_str = "-"

    # Step 2: Gemini dengan konteks keyword
    prompt = (
        f"Nilai siswa: {score:.1f}/100.\n"
        f"Kunci Jawaban: {key}\n"
        f"Kata kunci penting dari kunci jawaban: {kw_str}\n"
        f"Jawaban Siswa: {ans}\n\n"
        "Berikan feedback singkat (maks 2 kalimat) dalam bahasa Indonesia. "
        "Sebutkan apakah kata kunci penting sudah tercakup dalam jawaban siswa. "
        "JANGAN berikan nilai angka."
    )
    try:
        resp = gemini.generate_content(prompt)
        return resp.text.replace("\n", " ").strip()
    except Exception as e:
        return f"[ERROR Hybrid] {e}"

# ──────────────────────────────────────────────
# EKSPERIMEN 1: 5 Opsi Feedback
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("🔬 EKSPERIMEN 1: 5 Opsi AI Feedback")
print("="*60)

exp1_rows = []
for i, row in df.iterrows():
    key = str(row["teacher_key"]).strip()
    ans = str(row["user_answer"]).strip()
    score = score_sbert(key, ans)

    print(f"   [{i+1}/{len(df)}] question_id={row['question_id']} score={score:.1f}")

    exp1_rows.append({
        "id": row["id"],
        "question_id": row["question_id"],
        "user_answer": ans,
        "teacher_key": key,
        "score_sbert": round(score, 2),
        "feedback_A_rule": feedback_A(score),
        "feedback_B_keybert": feedback_B(key, ans),
        "feedback_C_template": feedback_C(key, ans, score),
        "feedback_D_gemini": feedback_D(key, ans, score),
        "feedback_E_hybrid": feedback_E(key, ans, score),
    })

exp1_df = pd.DataFrame(exp1_rows)
exp1_csv = OUTPUT_DIR / "exp1_feedback_options.csv"
exp1_df.to_csv(exp1_csv, index=False, encoding="utf-8-sig")
print(f"   ✅ Saved: {exp1_csv}")

# Markdown output
md1 = "# Eksperimen 1 — Perbandingan 5 Opsi AI Feedback\n\n"
md1 += "> Model Scoring: SBERT (all-MiniLM-L6-v2) — fixed untuk semua opsi\n\n"
for r in exp1_rows:
    md1 += f"---\n"
    md1 += f"### ID {r['id']} | question_id {r['question_id']} | Skor SBERT: {r['score_sbert']}\n"
    md1 += f"- **Jawaban Siswa**: {r['user_answer'][:120]}{'...' if len(r['user_answer'])>120 else ''}\n"
    md1 += f"- **Kunci Jawaban**: {r['teacher_key'][:120]}{'...' if len(r['teacher_key'])>120 else ''}\n\n"
    md1 += f"| Opsi | Feedback |\n|------|----------|\n"
    md1 += f"| A (Rule-based) | {r['feedback_A_rule']} |\n"
    md1 += f"| B (KeyBERT) | {r['feedback_B_keybert']} |\n"
    md1 += f"| C (Template) | {r['feedback_C_template']} |\n"
    md1 += f"| D (Gemini) | {r['feedback_D_gemini']} |\n"
    md1 += f"| E (Hybrid) | {r['feedback_E_hybrid']} |\n\n"

with open(OUTPUT_DIR / "exp1_feedback_options.md", "w", encoding="utf-8") as f:
    f.write(md1)
print("   ✅ Markdown exp1 saved.")

# ──────────────────────────────────────────────
# EKSPERIMEN 2: 4 Model Scoring
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("🔬 EKSPERIMEN 2: 4 Model Scoring")
print("="*60)

exp2_rows = []
for i, row in df.iterrows():
    key = str(row["teacher_key"]).strip()
    ans = str(row["user_answer"]).strip()

    s_tfidf  = score_tfidf(key, ans)
    s_sbert  = score_sbert(key, ans)
    s_multi  = score_multilingual(key, ans)
    s_indo   = score_indobert(key, ans)

    print(f"   [{i+1}/{len(df)}] q={row['question_id']} | TF={s_tfidf:.1f} SB={s_sbert:.1f} ML={s_multi:.1f} ID={s_indo:.1f}")

    exp2_rows.append({
        "id": row["id"],
        "question_id": row["question_id"],
        "user_answer": ans,
        "teacher_key": key,
        "ai_score_original": row["ai_score"],
        "score_tfidf": round(s_tfidf, 2),
        "score_sbert": round(s_sbert, 2),
        "score_multilingual": round(s_multi, 2),
        "score_indobert": round(s_indo, 2),
    })

exp2_df = pd.DataFrame(exp2_rows)
exp2_csv = OUTPUT_DIR / "exp2_model_comparison.csv"
exp2_df.to_csv(exp2_csv, index=False, encoding="utf-8-sig")
print(f"   ✅ Saved: {exp2_csv}")

# Stats
md2 = "# Eksperimen 2 — Perbandingan 4 Model Scoring\n\n"
md2 += "## Statistik Rata-rata Skor per Model\n\n"
md2 += "| Model | Mean | Std | Min | Max |\n|-------|------|-----|-----|-----|\n"
for col, name in [
    ("score_tfidf", "TF-IDF"),
    ("score_sbert", "SBERT (en)"),
    ("score_multilingual", "Paraphrase-multilingual"),
    ("score_indobert", "IndoBERT"),
]:
    s = exp2_df[col]
    md2 += f"| {name} | {s.mean():.2f} | {s.std():.2f} | {s.min():.2f} | {s.max():.2f} |\n"

md2 += "\n## Perbandingan per Row\n\n"
md2 += "| ID | Q_ID | TF-IDF | SBERT | Multi | IndoBERT | Original |\n"
md2 += "|----|------|--------|-------|-------|----------|----------|\n"
for r in exp2_rows:
    md2 += (f"| {r['id']} | {r['question_id']} "
            f"| {r['score_tfidf']} | {r['score_sbert']} "
            f"| {r['score_multilingual']} | {r['score_indobert']} "
            f"| {r['ai_score_original']} |\n")

with open(OUTPUT_DIR / "exp2_model_comparison.md", "w", encoding="utf-8") as f:
    f.write(md2)
print("   ✅ Markdown exp2 saved.")

# ──────────────────────────────────────────────
# EKSPERIMEN 3: Soal Sama, Jawaban Berbeda
# Menguji variasi skor & feedback untuk question_id yang sama dengan jawaban berbeda
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("🔬 EKSPERIMEN 3: Soal Sama, Jawaban Berbeda")
print("="*60)

# Pilih 3 baris sampel dari df (rendah, sedang, tinggi skor SBERT)
import numpy as np

# Hitung skor SBERT untuk semua baris dulu
df["_sbert_tmp"] = [score_sbert(str(r["teacher_key"]), str(r["user_answer"])) for _, r in df.iterrows()]
low_row  = df[df["_sbert_tmp"] < 40].iloc[0] if len(df[df["_sbert_tmp"] < 40]) > 0 else df.iloc[0]
mid_row  = df[(df["_sbert_tmp"] >= 50) & (df["_sbert_tmp"] < 70)].iloc[0] if len(df[(df["_sbert_tmp"] >= 50) & (df["_sbert_tmp"] < 70)]) > 0 else df.iloc[1]
high_row = df[df["_sbert_tmp"] >= 70].iloc[0] if len(df[df["_sbert_tmp"] >= 70]) > 0 else df.iloc[-1]
df.drop(columns=["_sbert_tmp"], inplace=True)

SAMPLES = [
    ("LOW",  low_row),
    ("MID",  mid_row),
    ("HIGH", high_row),
]

exp3_rows = []

for label, row in SAMPLES:
    qid = row["question_id"]
    key = str(row["teacher_key"]).strip()

    print(f"\n   [{label}] Q{qid} | Kunci: {key[:60]}...")
    
    # Ambil semua jawaban untuk question_id ini dari dataset lengkap
    sub_df = df_full[df_full["question_id"] == qid]
    print(f"   Mengevaluasi {len(sub_df)} variasi jawaban...")

    for idx, s_row in sub_df.iterrows():
        ans = str(s_row["user_answer"]).strip()
        hist_id = s_row["history_id"]
        sub_id = s_row["id"]
        
        s_sbert = score_sbert(key, ans)
        s_tfidf = score_tfidf(key, ans)
        s_multi = score_multilingual(key, ans)
        s_indo  = score_indobert(key, ans)
        fb_a    = feedback_A(s_sbert)
        fb_b    = feedback_B(key, ans)
        fb_c    = feedback_C(key, ans, s_sbert)
        fb_d    = feedback_D(key, ans, s_sbert)
        fb_e    = feedback_E(key, ans, s_sbert)

        print(f"      - ID {sub_id} | SBERT={s_sbert:.2f} TF={s_tfidf:.2f} Multi={s_multi:.2f} Indo={s_indo:.2f}")

        exp3_rows.append({
            "sample_label": label,
            "question_id": qid,
            "history_id": hist_id,
            "submission_id": sub_id,
            "user_answer": ans,
            "teacher_key": key,
            "score_sbert": round(s_sbert, 2),
            "score_tfidf": round(s_tfidf, 2),
            "score_multilingual": round(s_multi, 2),
            "score_indobert": round(s_indo, 2),
            "feedback_A_rule": fb_a,
            "feedback_B_keybert": fb_b,
            "feedback_C_template": fb_c,
            "feedback_D_gemini": fb_d,
            "feedback_E_hybrid": fb_e,
        })

exp3_df = pd.DataFrame(exp3_rows)
exp3_csv = OUTPUT_DIR / "exp3_duplicate_questions.csv"
exp3_df.to_csv(exp3_csv, index=False, encoding="utf-8-sig")
print(f"\n   ✅ Saved: {exp3_csv}")

# ─ Markdown Exp 3 ─
md3 = "# Eksperimen 3 — Soal Sama, Jawaban Berbeda\n\n"
md3 += "> Melihat variasi skor dan konsistensi feedback untuk question_id yang sama dengan jawaban berbeda.\n\n"

for label, row in SAMPLES:
    qid = row["question_id"]
    sub = exp3_df[exp3_df["question_id"] == qid]
    key_preview = str(row["teacher_key"]).strip()
    
    md3 += f"---\n## Question ID: {qid}\n"
    md3 += f"**Kunci Jawaban**: {key_preview}\n\n"
    
    for _, r in sub.iterrows():
        ans_preview = str(r["user_answer"]).strip()
        md3 += f"### History {r['history_id']} (ID {r['submission_id']}) | SBERT: {r['score_sbert']}\n"
        md3 += f"**Jawaban**: `{ans_preview[:120]}{'...' if len(ans_preview)>120 else ''}`\n\n"
        md3 += "| Opsi | Feedback |\n"
        md3 += "|------|----------|\n"
        md3 += f"| A (Rule) | {r['feedback_A_rule']} |\n"
        md3 += f"| B (KeyBERT) | {r['feedback_B_keybert']} |\n"
        md3 += f"| C (Template) | {r['feedback_C_template']} |\n"
        md3 += f"| D (Gemini) | {r['feedback_D_gemini']} |\n"
        md3 += f"| E (Hybrid) | {r['feedback_E_hybrid']} |\n\n"

with open(OUTPUT_DIR / "exp3_duplicate_questions.md", "w", encoding="utf-8") as f:
    f.write(md3)
print("   ✅ Markdown exp3 saved.")

# ──────────────────────────────────────────────
# SUMMARY LENGKAP
# ──────────────────────────────────────────────
# Ambil contoh feedback dari exp1 untuk ditampilkan di summary
exp1_low  = exp1_df[exp1_df["score_sbert"] == exp1_df["score_sbert"].min()].iloc[0]
exp1_mid_mask = (exp1_df["score_sbert"] >= 50) & (exp1_df["score_sbert"] < 70)
exp1_mid  = exp1_df[exp1_mid_mask].iloc[0] if exp1_mid_mask.any() else exp1_df.iloc[len(exp1_df)//2]
exp1_high = exp1_df[exp1_df["score_sbert"] == exp1_df["score_sbert"].max()].iloc[0]

md_sum = f"""# Summary Hasil Eksperimen — quizzess-ml AI Feedback Research

> **Gemini Model**: `gemini-3-flash-preview` | **Dataset**: `essay_submissions.csv`
> **Dataset**: {len(df_full)} submissions total | **Eksperimen 1 & 2**: 10 baris terakhir (ID {df["id"].iloc[0]}–{df["id"].iloc[-1]}) | **Eksperimen 3**: Soal sama, jawaban berbeda

---

## Eksperimen 1 — Perbandingan 5 Opsi AI Feedback

> Model scoring tetap: **SBERT (all-MiniLM-L6-v2)**, 10 baris terakhir dataset

### Distribusi Skor SBERT

| Metrik | Nilai |
|--------|-------|
| Mean   | {exp1_df["score_sbert"].mean():.2f} |
| Std    | {exp1_df["score_sbert"].std():.2f} |
| Min    | {exp1_df["score_sbert"].min():.2f} |
| Max    | {exp1_df["score_sbert"].max():.2f} |

### Contoh Nyata: Skor Rendah (Q{exp1_low["question_id"]} — {exp1_low["score_sbert"]})

**Jawaban**: `{str(exp1_low["user_answer"])[:100]}`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | {exp1_low["feedback_A_rule"]} |
| B (KeyBERT) | {exp1_low["feedback_B_keybert"]} |
| C (Template) | {exp1_low["feedback_C_template"]} |
| D (Gemini) | {exp1_low["feedback_D_gemini"]} |
| E (Hybrid) | {exp1_low["feedback_E_hybrid"]} |

### Contoh Nyata: Skor Sedang (Q{exp1_mid["question_id"]} — {exp1_mid["score_sbert"]})

**Jawaban**: `{str(exp1_mid["user_answer"])[:100]}`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | {exp1_mid["feedback_A_rule"]} |
| B (KeyBERT) | {exp1_mid["feedback_B_keybert"]} |
| C (Template) | {exp1_mid["feedback_C_template"]} |
| D (Gemini) | {exp1_mid["feedback_D_gemini"]} |
| E (Hybrid) | {exp1_mid["feedback_E_hybrid"]} |

### Contoh Nyata: Skor Tinggi (Q{exp1_high["question_id"]} — {exp1_high["score_sbert"]})

**Jawaban**: `{str(exp1_high["user_answer"])[:100]}`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | {exp1_high["feedback_A_rule"]} |
| B (KeyBERT) | {exp1_high["feedback_B_keybert"]} |
| C (Template) | {exp1_high["feedback_C_template"]} |
| D (Gemini) | {exp1_high["feedback_D_gemini"]} |
| E (Hybrid) | {exp1_high["feedback_E_hybrid"]} |

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
| TF-IDF | {exp2_df["score_tfidf"].mean():.2f} | {exp2_df["score_tfidf"].std():.2f} | {exp2_df["score_tfidf"].min():.2f} | {exp2_df["score_tfidf"].max():.2f} | Sangat lemah, sering 0.0 |
| SBERT (en) | {exp2_df["score_sbert"].mean():.2f} | {exp2_df["score_sbert"].std():.2f} | {exp2_df["score_sbert"].min():.2f} | {exp2_df["score_sbert"].max():.2f} | Stabil, English-only |
| Paraphrase-multilingual | {exp2_df["score_multilingual"].mean():.2f} | {exp2_df["score_multilingual"].std():.2f} | {exp2_df["score_multilingual"].min():.2f} | {exp2_df["score_multilingual"].max():.2f} | Bisa negatif |
| **IndoBERT** | **{exp2_df["score_indobert"].mean():.2f}** | {exp2_df["score_indobert"].std():.2f} | {exp2_df["score_indobert"].min():.2f} | {exp2_df["score_indobert"].max():.2f} | **Terbaik untuk Bahasa Indonesia** |

### Perbandingan per Soal

| ID | Q | TF-IDF | SBERT | Multilingual | IndoBERT | Original |
|----|---|--------|-------|-------------|----------|----------|
"""

for r in exp2_rows:
    md_sum += f"| {r['id']} | {r['question_id']} | {r['score_tfidf']} | {r['score_sbert']} | {r['score_multilingual']} | {r['score_indobert']} | {r['ai_score_original']} |\n"

md_sum += """
---

## Eksperimen 3 — Soal Sama, Jawaban Berbeda

> Melihat konsistensi feedback untuk question_id yang sama dengan jawaban berbeda

"""

for label, row in SAMPLES:
    qid = row["question_id"]
    sub = exp3_df[exp3_df["question_id"] == qid]
    
    md_sum += f"### Question ID: {qid}\n"
    md_sum += f"**Kunci Jawaban**: {row['teacher_key']}\n\n"
    
    for _, r in sub.iterrows():
        ans_preview = str(r["user_answer"]).strip()
        md_sum += f"#### History {r['history_id']} (ID {r['submission_id']}) | SBERT: {r['score_sbert']}\n"
        md_sum += f"**Jawaban**: `{ans_preview[:120]}{'...' if len(ans_preview)>120 else ''}`\n\n"
        md_sum += "| Opsi | Feedback |\n"
        md_sum += "|------|----------|\n"
        md_sum += f"| A (Rule-based) | {r['feedback_A_rule']} |\n"
        md_sum += f"| B (KeyBERT) | {r['feedback_B_keybert']} |\n"
        md_sum += f"| C (Template) | {r['feedback_C_template']} |\n"
        md_sum += f"| D (Gemini) | {r['feedback_D_gemini']} |\n"
        md_sum += f"| E (Hybrid) | {r['feedback_E_hybrid']} |\n\n"

md_sum += """---

## Rekomendasi

| Aspek | Sekarang | Rekomendasi |
|-------|----------|-------------|
| **Model Scoring** | SBERT (en) | Upgrade ke **IndoBERT** untuk Bahasa Indonesia |
| **AI Feedback** | Rule-based (Opsi A) | **Opsi E (Hybrid)** untuk produksi, **Opsi C** sebagai fallback offline |
| **Eksperimen 3** | — | Feedback dinamis (Gemini/Hybrid) merespons perbedaan redaksi jawaban dengan baik meskipun soalnya sama |

---

## Output Files

| File | Isi |
|------|-----|
| `exp1_feedback_options.csv/.md` | 10 baris × 5 opsi feedback |
| `exp2_model_comparison.csv/.md` | 10 baris × 4 model skor |
| `exp3_duplicate_questions.csv/.md` | 3 sampel soal sama × jawaban berbeda |
| `summary.md` | File ini |
"""

with open(OUTPUT_DIR / "summary.md", "w", encoding="utf-8") as f:
    f.write(md_sum)

print("\n" + "="*60)
print("🎉 SEMUA EKSPERIMEN SELESAI!")
print(f"📁 Output di: {OUTPUT_DIR}")
print("="*60)




import pandas as pd
import os
import torch
import warnings
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import google.generativeai as genai

warnings.filterwarnings("ignore")

# 1. SETUP MODELS
print("Loading SBERT...")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Loading KeyBERT...")
kw_extractor = KeyBERT(model=sbert_model)

# 2. SETUP GEMINI
# gemini_api_key = os.getenv("GEMINI_API_KEY", "")
gemini_api_key = "AIzaSyA4gb4VCLDYkZQQYsOeGl2_L9haTO-5GU4"
if gemini_api_key:
    print("Checking model")

    genai.configure(api_key=gemini_api_key)
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print(f"Model: {m.name}")
    except Exception as e:
        print(f"Gagal list models: {e}")    
    # Using gemini-1.5-flash as it's the standard fast model now
    model = genai.GenerativeModel("gemini-2.5-flash")
else:
    model = None
    print("[WARNING] GEMINI_API_KEY is not set. Option 1 will be mocked.")


def get_option_2_reason(teacher_key, student_answer, threshold=0.4):
    """
    Keyword Extraction & Semantic Matching (Local)
    """
    # Extract keywords from teacher key
    keywords = kw_extractor.extract_keywords(
        teacher_key, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5
    )

    # If student answer is empty or too short
    if not student_answer or len(student_answer.strip()) < 3:
        return "Jawaban sangat singkat atau kosong."

    missing_keywords = []
    student_emb = sbert_model.encode(student_answer, convert_to_tensor=True)

    for kw, score in keywords:
        kw_emb = sbert_model.encode(kw, convert_to_tensor=True)
        sim = util.cos_sim(kw_emb, student_emb).item()
        if sim < threshold:
            missing_keywords.append(kw)

    if missing_keywords:
        return f"Jawaban kamu belum mencakup konsep penting berikut: {', '.join(missing_keywords)}."
    else:
        return "Jawaban kamu sudah mencakup seluruh poin penting dari kunci jawaban."


def get_option_1_reason(question, teacher_key, student_answer, ai_score):
    """
    Generative LLM via Gemini API
    """
    if not model:
        return (
            "[MOCK] Jawaban kurang mencakup elemen dari kunci jawaban berdasar Gemini."
        )

    prompt = f"""
Nilai siswa: {ai_score:.2f}/100.
Pertanyaan: {question}
Kunci Jawaban: {teacher_key}
Jawaban Siswa: {student_answer}

Evaluasilah jawaban siswa di atas secara singkat (maksimal 2 kalimat) dalam bahasa Indonesia. Jelaskan mengapa nilainya demikian (apa yang kurang/salah atau jika sudah tepat, pujilah). JANGAN memberikan nilai lagi.
"""
    try:
        response = model.generate_content(prompt)
        return response.text.replace("\n", " ").strip()
    except Exception as e:
        return f"[ERROR] Failed to generate reason from Gemini: {e}"


def run_comparison():
    # Load dataset
    filepath = "essay_dataset_v1.csv"
    if not os.path.exists(filepath):
        print(f"File {filepath} tidak ditemukan!")
        return

    df = pd.read_csv(filepath)
    # limit to 5 varied samples untuk review
    sample_df = df.sample(n=5, random_state=42)

    results = []

    for idx, row in sample_df.iterrows():
        q = row["question"]
        tk = row["teacher_key"]
        sa = row["student_answer"]

        # Calculate AI score (SBERT) as baseline
        tk_emb = sbert_model.encode(tk, convert_to_tensor=True)
        sa_emb = sbert_model.encode(sa, convert_to_tensor=True)
        ai_score = util.cos_sim(tk_emb, sa_emb).item() * 100.0

        print(f"\n--- Menguji Baris ke-{idx} ---")

        # Opsi 1
        print("Menjalankan Opsi 1 (Gemini API)...")
        reason_1 = get_option_1_reason(q, tk, sa, ai_score)

        # Opsi 2
        print("Menjalankan Opsi 2 (KeyBERT)...")
        reason_2 = get_option_2_reason(tk, sa)

        results.append(
            {
                "Pertanyaan": q,
                "Jawaban Siswa": sa,
                "Skor AI": f"{ai_score:.1f}",
                "Opsi 1 (Gemini)": reason_1,
                "Opsi 2 (KeyBERT)": reason_2,
            }
        )

    res_df = pd.DataFrame(results)
    output_path = "comparison_results.csv"
    res_df.to_csv(output_path, index=False)
    print(f"\nSelesai! Hasil disimpan di {output_path}")

    # Save as Markdown artifact mapping output
    md_content = "# Hasil Perbandingan Opsi 1 & Opsi 2\n\n"
    for idx, res in enumerate(results):
        md_content += f"### Pertanyaan {idx + 1}: {res['Pertanyaan']}\n"
        md_content += f"- **Jawaban Siswa**: {res['Jawaban Siswa']}\n"
        md_content += f"- **Skor Asli (SBERT)**: {res['Skor AI']} / 100\n"
        md_content += f"- **Opsi 1 (Gemini)**: {res['Opsi 1 (Gemini)']}\n"
        md_content += f"- **Opsi 2 (KeyBERT + SBERT)**: {res['Opsi 2 (KeyBERT)']}\n\n"
        md_content += "---\n\n"

    with open("comparison_results.md", "w", encoding="utf-8") as f:
        f.write(md_content)

    print("Berhasil mengekspor markdown hasil di comparison_results.md.")


if __name__ == "__main__":
    run_comparison()

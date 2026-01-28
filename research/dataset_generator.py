import pandas as pd
import random

# Definisi Soal & Kunci Jawaban (Biologi Konteks)
topics = [
    {
        "q": "Jelaskan fungsi utama jantung.",
        "k": "Jantung berfungsi untuk memompa darah ke seluruh tubuh melalui pembuluh darah, membawa oksigen dan nutrisi."
    },
    {
        "q": "Apa yang dimaksud dengan fotosintesis?",
        "k": "Fotosintesis adalah proses di mana tumbuhan hijau mengubah energi cahaya matahari menjadi energi kimia dalam bentuk glukosa/makanan, menggunakan air dan karbondioksida."
    },
    {
        "q": "Sebutkan fungsi sel darah merah (eritrosit).",
        "k": "Fungsi utama sel darah merah adalah mengangkut oksigen dari paru-paru ke seluruh jaringan tubuh dan membawa kembali karbondioksida ke paru-paru."
    },
    {
        "q": "Jelaskan perbedaan pembuluh Arteri dan Vena.",
        "k": "Arteri mengalirkan darah keluar dari jantung (biasanya kaya oksigen), dindingnya tebal dan elastis. Vena mengalirkan darah kembali ke jantung (biasanya kaya karbondioksida), dindingnya lebih tipis dan memiliki katup."
    },
    {
        "q": "Apa fungsi mitokondria dalam sel?",
        "k": "Mitokondria berfungsi sebagai tempat respirasi sel untuk menghasilkan energi dalam bentuk ATP."
    }
]

# Variasi Jawaban Siswa (Dummy)
# Kategori: 
# A: Jawaban Tepat (Mirip Kunci)
# B: Jawaban Benar tapi Singkat
# C: Jawaban Benar tapi Bahasa Beda
# D: Jawaban Kurang Lengkap
# E: Jawaban Salah Total

data = []

for item in topics:
    # 1. Jawaban Tepat (A)
    data.append({
        "question": item["q"],
        "teacher_key": item["k"],
        "student_answer": item["k"], # Exact match
        "label_human": 100,
        "type": "exact"
    })
    
    # 2. Jawaban Benar Singkat (B)
    # Generate simple variations manually for demo
    if "Jantung" in item["q"]:
        ans = "Memompa darah ke tubuh."
    elif "fotosintesis" in item["q"]:
        ans = "Tumbuhan bikin makanan pakai matahari."
    elif "sel darah merah" in item["q"]:
        ans = "Bawa oksigen."
    elif "Arteri" in item["q"]:
        ans = "Arteri bawa darah dari jantung, Vena ke jantung."
    else:
        ans = "Menghasilkan energi ATP."
        
    data.append({
        "question": item["q"],
        "teacher_key": item["k"],
        "student_answer": ans,
        "label_human": 85,
        "type": "short_correct"
    })
    
    # 3. Jawaban Salah (E)
    wrong_ans = "Saya tidak tahu."
    if "Jantung" in item["q"]:
        wrong_ans = "Jantung untuk bernafas."
    elif "fotosintesis" in item["q"]:
        wrong_ans = "Proses pembakaran hutan."
    
    data.append({
        "question": item["q"],
        "teacher_key": item["k"],
        "student_answer": wrong_ans,
        "label_human": 0,
        "type": "wrong"
    })
    
    # 4. Jawaban Kinda Correct (C)
    # Variation
    data.append({
        "question": item["q"],
        "teacher_key": item["k"],
        "student_answer": item["k"].replace("berfungsi", "gunanya").replace("seluruh tubuh", "semua badan"),
        "label_human": 95,
        "type": "synonym"
    })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(current_dir, "essay_dataset_v1.csv")

df.to_csv(filename, index=False)
print(f"✅ Generated {len(df)} dummy rows in {filename}")

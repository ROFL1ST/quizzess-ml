# Summary Hasil Eksperimen — quizzess-ml AI Feedback Research

> **Gemini Model**: `gemini-3-flash-preview` | **Dataset**: `essay_submissions.csv`
> **Dataset**: 50 submissions total | **Eksperimen 1 & 2**: 10 baris terakhir (ID 94–103) | **Eksperimen 3**: Soal sama, jawaban berbeda

---

## Eksperimen 1 — Perbandingan 5 Opsi AI Feedback

> Model scoring tetap: **SBERT (all-MiniLM-L6-v2)**, 10 baris terakhir dataset

### Distribusi Skor SBERT

| Metrik | Nilai |
|--------|-------|
| Mean   | 63.96 |
| Std    | 12.82 |
| Min    | 37.81 |
| Max    | 82.89 |

### Contoh Nyata: Skor Rendah (Q334 — 37.81)

**Jawaban**: `dampaknya adalah kenaikan suhu bumi yang dapat mengakibatkan melelehnya es dan menaikkan ketinggian `

**Kunci Jawaban**: `Sinar ultraviolet (UV) masuk ke bumi tanpa filter, memicu kanker kulit, katarak, dan kerusakan ekosistem.`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Jawaban tidak relevan dengan kunci jawaban. |
| B (KeyBERT) | Jawaban belum mencakup konsep penting: sinar ultraviolet, ultraviolet uv, ultraviolet, tanpa filter. |
| C (Template) | Jawaban belum sesuai kunci jawaban (37.8%). Pelajari ulang materi dan coba jawab dengan kata-kata sendiri. |
| D (Gemini) | Jawaban siswa kurang tepat karena menjelaskan dampak pemanasan global, padahal kunci jawaban fokus pada dampak penipisan lapisan ozon (radiasi UV). Siswa perlu memperbaiki pemahaman mengenai perbedaan antara efek rumah kaca dan kerusakan lapisan ozon agar tidak tertukar dampaknya. |
| E (Hybrid) | Jawaban Anda belum mencakup kata kunci penting "sinar ultraviolet" karena Anda justru menjelaskan dampak pemanasan global secara umum. Fokuslah pada dampak spesifik dari paparan radiasi UV sesuai dengan kunci jawaban. |

### Contoh Nyata: Skor Sedang (Q336 — 67.27)

**Jawaban**: `karena proses pembakaran mesin melepaskan emisi gas beracun secara masif, seperti karbon monoksida (`

**Kunci Jawaban**: `Pembakaran bahan bakar fosil melepaskan gas berbahaya seperti karbon monoksida dan karbon dioksida ke atmosfer.`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Kurang tepat, perbaiki lagi konsepnya. |
| B (KeyBERT) | Jawaban belum mencakup konsep penting: ke atmosfer, bahan bakar. |
| C (Template) | Jawaban cukup baik (67.3% sesuai kunci). Coba tambahkan detail atau istilah yang lebih spesifik sesuai materi. |
| D (Gemini) | Jawaban siswa sebenarnya sangat lengkap dan akurat secara ilmiah, bahkan melampaui detail kunci jawaban dengan menyebutkan polutan spesifik lainnya. Nilai tersebut kemungkinan diberikan karena siswa tidak secara eksplisit menyebutkan "karbon dioksida" (CO2) dan "bahan bakar fosil" yang merupakan kata kunci utama dalam kunci jawaban. |
| E (Hybrid) | Jawaban siswa telah mencakup kata kunci "melepaskan gas" dan "gas berbahaya" melalui penjelasan yang sangat detail dan akurat. Meskipun frasa "ke atmosfer" tidak tertulis secara eksplisit, penjelasan mengenai dampak lingkungan sudah mencakup konteks tersebut dengan sangat baik. |

### Contoh Nyata: Skor Tinggi (Q327 — 82.89)

**Jawaban**: `fenomena peningkatan suhu rata-rata atmosfer, lautan, dan daratan Bumi secara bertahap dan signifika`

**Kunci Jawaban**: `Peningkatan suhu rata-rata atmosfer, laut, dan daratan bumi secara global.`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Bagus, namun ada sedikit perbedaan istilah. |
| B (KeyBERT) | Jawaban sudah mencakup seluruh poin penting dari kunci jawaban. |
| C (Template) | Jawaban cukup baik (82.9% sesuai kunci). Coba tambahkan detail atau istilah yang lebih spesifik sesuai materi. |
| D (Gemini) | Jawaban siswa sudah sangat akurat dan lebih komprehensif karena menambahkan aspek "jangka panjang" dan "bertahap" yang melengkapi definisi tersebut. Nilai tersebut sangat baik, namun mungkin sedikit berkurang karena tidak mencantumkan kata "global" secara eksplisit sebagaimana tertera pada kunci jawaban. |
| E (Hybrid) | Jawaban siswa sangat akurat dan telah mencakup seluruh kata kunci penting secara lengkap. Penjelasan mengenai peningkatan suhu pada atmosfer, laut, dan daratan Bumi sudah tersampaikan dengan sangat baik. |

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
| TF-IDF | 25.65 | 14.15 | 6.98 | 58.83 | Sangat lemah, sering 0.0 |
| SBERT (en) | 63.96 | 12.82 | 37.81 | 82.89 | Stabil, English-only |
| Paraphrase-multilingual | 69.92 | 18.69 | 33.26 | 90.57 | Bisa negatif |
| **IndoBERT** | **71.43** | 10.69 | 52.67 | 91.01 | **Terbaik untuk Bahasa Indonesia** |

### Perbandingan per Soal

| ID | Q | TF-IDF | SBERT | Multilingual | IndoBERT | Original |
|----|---|--------|-------|-------------|----------|----------|
| 94 | 334 | 6.98 | 37.81 | 33.26 | 52.67 | 37.81 |
| 95 | 335 | 19.05 | 76.05 | 52.81 | 69.17 | 76.05 |
| 96 | 336 | 21.83 | 67.27 | 84.41 | 73.29 | 67.27 |
| 97 | 327 | 58.83 | 82.89 | 90.57 | 91.01 | 82.89 |
| 98 | 328 | 23.17 | 58.45 | 62.18 | 61.88 | 58.45 |
| 99 | 331 | 33.25 | 68.88 | 88.2 | 73.57 | 68.88 |
| 100 | 332 | 28.67 | 71.36 | 79.84 | 80.45 | 71.36 |
| 101 | 329 | 30.15 | 66.8 | 83.57 | 79.07 | 66.8 |
| 102 | 330 | 22.88 | 53.68 | 68.18 | 65.62 | 53.68 |
| 103 | 333 | 11.7 | 56.41 | 56.14 | 67.58 | 56.41 |

---

## Eksperimen 3 — Soal Sama, Jawaban Berbeda

> Melihat konsistensi feedback untuk question_id yang sama dengan jawaban berbeda

### Question ID: 334
**Kunci Jawaban**: Sinar ultraviolet (UV) masuk ke bumi tanpa filter, memicu kanker kulit, katarak, dan kerusakan ekosistem.

#### History 633 (ID 61) | SBERT: 5.08
**Jawaban**: `Pppp`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Jawaban tidak relevan dengan kunci jawaban. |
| B (KeyBERT) | Jawaban belum mencakup konsep penting: sinar ultraviolet, ultraviolet uv, ultraviolet, tanpa filter. |
| C (Template) | Jawaban belum sesuai kunci jawaban (5.1%). Pelajari ulang materi dan coba jawab dengan kata-kata sendiri. |
| D (Gemini) | Jawaban tidak relevan karena hanya berisi teks asal yang tidak berkaitan dengan materi dampak sinar ultraviolet. Siswa perlu mempelajari kembali materi tersebut dan memberikan penjelasan logis yang sesuai dengan pertanyaan. |
| E (Hybrid) | Jawaban Anda sama sekali tidak mencakup kata kunci penting seperti "sinar ultraviolet". Mohon berikan jawaban yang relevan dan sesuai dengan topik pembahasan. |

#### History 634 (ID 67) | SBERT: 31.22
**Jawaban**: `terjadinya pemasan global dan mencairnya es di kutub`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Jawaban tidak relevan dengan kunci jawaban. |
| B (KeyBERT) | Jawaban belum mencakup konsep penting: sinar ultraviolet, ultraviolet uv, ultraviolet, tanpa filter. |
| C (Template) | Jawaban belum sesuai kunci jawaban (31.2%). Pelajari ulang materi dan coba jawab dengan kata-kata sendiri. |
| D (Gemini) | Jawaban siswa kurang tepat karena mencampuradukkan dampak penipisan lapisan ozon dengan fenomena pemanasan global (efek rumah kaca). Siswa perlu memperbaiki pemahaman mengenai dampak spesifik radiasi ultraviolet terhadap kesehatan dan ekosistem agar relevan dengan kunci jawaban. |
| E (Hybrid) | Jawaban Anda belum mencakup kata kunci penting seperti "sinar ultraviolet". Anda masih tertukar antara dampak penipisan lapisan ozon dengan fenomena pemanasan global. |

#### History 635 (ID 82) | SBERT: 63.54
**Jawaban**: `Jika lapisan ozon terus menipis, dampaknya sangat serius bagi Bumi, terutama karena ozon berperan sebagai pelindung alam...`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Kurang tepat, perbaiki lagi konsepnya. |
| B (KeyBERT) | Jawaban belum mencakup konsep penting: tanpa filter. |
| C (Template) | Jawaban kurang tepat (63.5% sesuai kunci). Periksa kembali konsep utama dan gunakan kata kunci yang relevan. |
| D (Gemini) | Jawaban siswa sebenarnya sudah sangat lengkap dan akurat karena mencakup seluruh poin kunci (kanker kulit, katarak, dan ekosistem) dengan penjelasan yang lebih mendalam. Nilai tersebut kemungkinan rendah karena jawaban terlalu panjang atau tidak langsung pada inti kunci jawaban, padahal secara substansi jawaban ini sudah sangat baik. |
| E (Hybrid) | Jawaban siswa telah mencakup kata kunci penting mengenai radiasi ultraviolet (UV) dengan sangat baik. Seluruh dampak utama yang diminta seperti risiko kanker kulit, katarak, dan kerusakan ekosistem juga telah dijelaskan secara mendalam dan akurat. |

#### History 637 (ID 86) | SBERT: 22.22
**Jawaban**: `bumi akan memanas`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Jawaban tidak relevan dengan kunci jawaban. |
| B (KeyBERT) | Jawaban belum mencakup konsep penting: sinar ultraviolet, ultraviolet uv, ultraviolet, tanpa filter. |
| C (Template) | Jawaban belum sesuai kunci jawaban (22.2%). Pelajari ulang materi dan coba jawab dengan kata-kata sendiri. |
| D (Gemini) | Jawaban kurang tepat karena "bumi memanas" lebih merujuk pada fenomena pemanasan global, sementara kunci jawaban menekankan pada dampak spesifik radiasi ultraviolet terhadap kesehatan dan ekosistem. Siswa perlu mempelajari lebih lanjut perbedaan antara penipisan lapisan ozon dengan efek rumah kaca agar dapat menyebutkan dampak biologis yang lebih relevan. |
| E (Hybrid) | Jawaban Anda belum mencakup kata kunci penting seperti "sinar ultraviolet" atau "UV". Fokus jawaban masih kurang tepat karena lebih membahas pemanasan suhu daripada dampak spesifik radiasi ultraviolet terhadap kesehatan dan ekosistem. |

#### History 638 (ID 94) | SBERT: 37.81
**Jawaban**: `dampaknya adalah kenaikan suhu bumi yang dapat mengakibatkan melelehnya es dan menaikkan ketinggian permukaan air laut`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Jawaban tidak relevan dengan kunci jawaban. |
| B (KeyBERT) | Jawaban belum mencakup konsep penting: sinar ultraviolet, ultraviolet uv, ultraviolet, tanpa filter. |
| C (Template) | Jawaban belum sesuai kunci jawaban (37.8%). Pelajari ulang materi dan coba jawab dengan kata-kata sendiri. |
| D (Gemini) | Jawaban siswa kurang tepat karena menjelaskan dampak pemanasan global, sementara kunci jawaban merujuk pada dampak spesifik radiasi UV akibat penipisan lapisan ozon. Siswa perlu memperbaiki pemahaman dalam membedakan antara efek rumah kaca (kenaikan suhu) dengan bahaya radiasi ultraviolet bagi kesehatan dan ekosistem. |
| E (Hybrid) | Jawaban Anda belum mencakup kata kunci penting "sinar ultraviolet". Anda menjelaskan dampak pemanasan global, sementara kunci jawaban memfokuskan pada dampak kesehatan dan ekosistem akibat radiasi UV. |

### Question ID: 336
**Kunci Jawaban**: Pembakaran bahan bakar fosil melepaskan gas berbahaya seperti karbon monoksida dan karbon dioksida ke atmosfer.

#### History 633 (ID 57) | SBERT: 15.18
**Jawaban**: `D`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Jawaban tidak relevan dengan kunci jawaban. |
| B (KeyBERT) | Jawaban terlalu singkat atau kosong. |
| C (Template) | Jawaban terlalu singkat. Kemiripan dengan kunci jawaban hanya 15.2%. |
| D (Gemini) | Jawaban siswa tidak tepat karena hanya menyertakan pilihan huruf tanpa memberikan penjelasan materi yang diminta oleh soal. Siswa perlu menguraikan secara lengkap jenis gas yang dilepaskan ke atmosfer untuk memenuhi kriteria jawaban yang benar. |
| E (Hybrid) | Jawaban siswa tidak mencakup kata kunci penting seperti gas berbahaya, melepaskan gas, atau ke atmosfer. Jawaban tersebut hanya berupa satu huruf dan belum memberikan penjelasan yang sesuai dengan kunci jawaban. |

#### History 634 (ID 73) | SBERT: 53.14
**Jawaban**: `dari karbon yang mereka keluarkan`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Kurang tepat, perbaiki lagi konsepnya. |
| B (KeyBERT) | Jawaban belum mencakup konsep penting: gas berbahaya, melepaskan gas, ke atmosfer. |
| C (Template) | Jawaban kurang tepat (53.1% sesuai kunci). Periksa kembali konsep utama dan gunakan kata kunci yang relevan. |
| D (Gemini) | Jawaban kurang lengkap karena hanya menyebutkan unsur karbon tanpa menjelaskan proses pembakaran bahan bakar fosil dan dampaknya bagi atmosfer. Siswa perlu merinci jenis gas spesifik (seperti karbon monoksida/dioksida) agar penjelasan lebih akurat dan menyeluruh sesuai kunci jawaban. |
| E (Hybrid) | Jawaban siswa belum mencakup kata kunci penting seperti "gas berbahaya" dan "ke atmosfer". Penjelasan mengenai pelepasan karbon tersebut masih kurang lengkap dan kurang spesifik dibandingkan kunci jawaban. |

#### History 635 (ID 77) | SBERT: 56.75
**Jawaban**: `Kendaraan fosil adalah sumber utama polusi udara di perkotaan karena mereka secara konstan melepaskan zat beracun ke uda...`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Kurang tepat, perbaiki lagi konsepnya. |
| B (KeyBERT) | Jawaban belum mencakup konsep penting: gas berbahaya, melepaskan gas, ke atmosfer. |
| C (Template) | Jawaban kurang tepat (56.8% sesuai kunci). Periksa kembali konsep utama dan gunakan kata kunci yang relevan. |
| D (Gemini) | Jawaban siswa sudah benar secara konsep mengenai dampak polusi, namun kurang spesifik karena tidak menyebutkan jenis gas yang dilepaskan secara mendetail. Sebaiknya sebutkan karbon monoksida dan karbon dioksida sesuai kunci jawaban agar penjelasan lebih akurat secara ilmiah. |
| E (Hybrid) | Jawaban Anda sudah mencakup semua poin dari kata kunci penting, meskipun menggunakan sinonim seperti "zat beracun" dan "ke udara". Secara keseluruhan, inti dari kunci jawaban telah tersampaikan dengan baik dan akurat. |

#### History 637 (ID 87) | SBERT: 63.1
**Jawaban**: `Kendaraan berbahan bakar fosil dianggap penyumbang polusi udara utama karena proses pembakaran bensin atau solar menghas...`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Kurang tepat, perbaiki lagi konsepnya. |
| B (KeyBERT) | Jawaban belum mencakup konsep penting: melepaskan gas, ke atmosfer, bahan bakar. |
| C (Template) | Jawaban kurang tepat (63.1% sesuai kunci). Periksa kembali konsep utama dan gunakan kata kunci yang relevan. |
| D (Gemini) | Jawaban siswa sebenarnya sudah sangat tepat dan jauh lebih komprehensif daripada kunci jawaban karena merinci jenis polutan serta dampak kesehatan secara mendalam. Nilai tersebut kemungkinan rendah karena adanya ketidaksesuaian kata kunci dengan sistem penilaian, padahal secara substansi jawaban sudah sangat lengkap dan benar. |
| E (Hybrid) | Jawaban siswa sudah sangat lengkap dan mencakup seluruh kata kunci penting, yaitu pelepasan gas berbahaya ke atmosfer (udara). Penjelasan yang diberikan bahkan lebih mendalam dengan menyertakan jenis gas spesifik serta dampaknya bagi kesehatan dan lingkungan. |

#### History 638 (ID 96) | SBERT: 67.27
**Jawaban**: `karena proses pembakaran mesin melepaskan emisi gas beracun secara masif, seperti karbon monoksida (CO), nitrogen oksida...`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Kurang tepat, perbaiki lagi konsepnya. |
| B (KeyBERT) | Jawaban belum mencakup konsep penting: ke atmosfer, bahan bakar. |
| C (Template) | Jawaban cukup baik (67.3% sesuai kunci). Coba tambahkan detail atau istilah yang lebih spesifik sesuai materi. |
| D (Gemini) | Jawaban siswa sebenarnya sangat komprehensif dan benar, namun nilai tersebut kurang maksimal karena belum menyebutkan kata kunci "karbon dioksida" secara eksplisit seperti pada kunci jawaban. Pastikan poin utama dari kunci jawaban tetap disertakan di samping penjelasan teknis tambahan yang sudah sangat baik tersebut. |
| E (Hybrid) | Jawaban siswa telah mencakup kata kunci "melepaskan gas" dan "gas berbahaya" dengan penjelasan yang sangat mendalam. Namun, kata kunci "ke atmosfer" belum disebutkan secara eksplisit dalam jawaban tersebut. |

### Question ID: 335
**Kunci Jawaban**: Mematikan keran air saat menyabuni tangan atau menyikat gigi, serta memperbaiki kebocoran pipa.

#### History 633 (ID 56) | SBERT: 17.25
**Jawaban**: `P`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Jawaban tidak relevan dengan kunci jawaban. |
| B (KeyBERT) | Jawaban terlalu singkat atau kosong. |
| C (Template) | Jawaban terlalu singkat. Kemiripan dengan kunci jawaban hanya 17.2%. |
| D (Gemini) | Jawaban siswa sangat tidak memadai karena hanya berupa satu huruf yang tidak memiliki kaitan dengan materi penghematan air. Siswa perlu menuliskan jawaban dalam bentuk kalimat yang lengkap dan relevan sesuai dengan instruksi pada kunci jawaban. |
| E (Hybrid) | Jawaban siswa sama sekali belum mencakup kata kunci penting yang diminta karena hanya berisi satu huruf. Harap berikan penjelasan yang lebih lengkap sesuai dengan isi kunci jawaban. |

#### History 634 (ID 72) | SBERT: 61.06
**Jawaban**: `menggunakan Air secukupnya`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Kurang tepat, perbaiki lagi konsepnya. |
| B (KeyBERT) | Jawaban belum mencakup konsep penting: atau menyikat, serta memperbaiki. |
| C (Template) | Jawaban kurang tepat (61.1% sesuai kunci). Periksa kembali konsep utama dan gunakan kata kunci yang relevan. |
| D (Gemini) | Jawaban siswa sudah benar secara prinsip, namun terlalu umum sehingga kurang mendetail. Siswa perlu memberikan contoh tindakan spesifik, seperti mematikan keran atau memperbaiki kebocoran, agar sesuai dengan permintaan kunci jawaban. |
| E (Hybrid) | Jawaban tersebut masih terlalu umum dan belum mencakup kata kunci penting seperti mematikan keran saat menyabuni atau menyikat gigi. Cobalah memberikan contoh tindakan yang lebih spesifik agar sesuai dengan kunci jawaban. |

#### History 635 (ID 76) | SBERT: 76.02
**Jawaban**: `Matikan keran saat menyikat gigi atau mencuci piring.`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Bagus, namun ada sedikit perbedaan istilah. |
| B (KeyBERT) | Jawaban sudah mencakup seluruh poin penting dari kunci jawaban. |
| C (Template) | Jawaban cukup baik (76.0% sesuai kunci). Coba tambahkan detail atau istilah yang lebih spesifik sesuai materi. |
| D (Gemini) | Jawaban siswa sudah benar dalam menyebutkan salah satu perilaku hemat air, namun belum mencantumkan poin penting mengenai perbaikan kebocoran pipa. Untuk mendapatkan nilai sempurna, siswa perlu melengkapi jawaban dengan tindakan teknis lainnya seperti yang tertera pada kunci jawaban. |
| E (Hybrid) | Sebagian besar kata kunci penting seperti mematikan keran dan menyikat gigi sudah tercakup dalam jawabanmu. Kamu telah menyebutkan contoh tindakan penghematan air dengan tepat. |

#### History 637 (ID 93) | SBERT: 67.31
**Jawaban**: `membuat mesin saring dri air kotor menjadi air bersih`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Kurang tepat, perbaiki lagi konsepnya. |
| B (KeyBERT) | Jawaban sudah mencakup seluruh poin penting dari kunci jawaban. |
| C (Template) | Jawaban cukup baik (67.3% sesuai kunci). Coba tambahkan detail atau istilah yang lebih spesifik sesuai materi. |
| D (Gemini) | Jawaban kurang tepat karena fokus pada pengolahan air limbah, bukan cara menghemat penggunaan air bersih sesuai kunci jawaban. Siswa perlu diperbaiki dengan memberikan contoh tindakan praktis untuk mengurangi pemborosan air secara langsung dalam aktivitas sehari-hari. |
| E (Hybrid) | Jawaban Anda belum mencakup kata kunci penting seperti "mematikan keran" atau "saat menyabuni" karena lebih fokus pada pengolahan air bersih. Sebaiknya arahkan jawaban pada tindakan penghematan air secara langsung sesuai dengan petunjuk kunci jawaban. |

#### History 638 (ID 95) | SBERT: 76.05
**Jawaban**: `mematikan keran air ketika sudah tidak digunakan`

| Opsi | Feedback |
|------|----------|
| A (Rule-based) | Bagus, namun ada sedikit perbedaan istilah. |
| B (KeyBERT) | Jawaban sudah mencakup seluruh poin penting dari kunci jawaban. |
| C (Template) | Jawaban cukup baik (76.0% sesuai kunci). Coba tambahkan detail atau istilah yang lebih spesifik sesuai materi. |
| D (Gemini) | Jawaban siswa sudah benar secara umum, namun kurang spesifik dalam memberikan contoh penghematan air saat kegiatan berlangsung (seperti saat menyabuni tangan). Untuk hasil maksimal, siswa perlu menambahkan tindakan preventif lain seperti memperbaiki kebocoran pipa sesuai dengan kunci jawaban. |
| E (Hybrid) | Kata kunci "mematikan keran" sudah tercakup dalam jawabanmu, namun kata kunci "saat menyabuni" atau "menyikat" belum disebutkan. Jawabanmu masih bersifat umum dan belum menyertakan konteks spesifik yang diminta oleh kunci jawaban. |

---

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

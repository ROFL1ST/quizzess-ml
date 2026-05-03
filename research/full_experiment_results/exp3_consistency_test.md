# Eksperimen 3 — Consistency Test

> Jawaban yang **sama** dijalankan grading sebanyak **3x** untuk menguji konsistensi skor dan feedback.

## Hasil Konsistensi per Sampel

---
### [LOW] Question 334
**Jawaban**: `dampaknya adalah kenaikan suhu bumi yang dapat mengakibatkan melelehnya es dan menaikkan ketinggian ...`

#### Skor per Run

| Run | SBERT | TF-IDF | Multilingual | IndoBERT |
|-----|-------|--------|-------------|----------|
| 1 | 37.814355 | 6.980811 | 33.257627 | 52.6676 |
| 2 | 37.814355 | 6.980811 | 33.257627 | 52.6676 |
| 3 | 37.814355 | 6.980811 | 33.257627 | 52.6676 |

#### Analisis Konsistensi

| Model | Min | Max | Std | Deterministik? |
|-------|-----|-----|-----|----------------|
| SBERT | 37.8144 | 37.8144 | 0.0 | ✅ YA |
| TF-IDF | 6.9808 | 6.9808 | 0.0 | ✅ YA |
| Multilingual | 33.2576 | 33.2576 | 0.0 | ✅ YA |
| IndoBERT | 52.6676 | 52.6676 | 0.0 | ✅ YA |

#### Variasi Feedback Gemini (Opsi D)

- **Run 1**: Jawaban kurang tepat karena siswa menjelaskan dampak pemanasan global, bukannya dampak penipisan lapisan ozon seperti yang diminta pada kunci jawaban. Siswa perlu belajar membedakan antara efek rumah kaca (kenaikan suhu) dengan kerusakan lapisan ozon yang berkaitan dengan radiasi sinar ultraviolet.
- **Run 2**: Jawaban siswa kurang tepat karena menjelaskan dampak pemanasan global, padahal kunci jawaban berfokus pada dampak penipisan lapisan ozon (radiasi UV). Siswa perlu lebih teliti dalam membedakan kedua fenomena lingkungan tersebut agar penjelasan yang diberikan relevan dengan pertanyaan.
- **Run 3**: Jawaban kurang tepat karena siswa menjelaskan dampak pemanasan global, padahal pertanyaan merujuk pada dampak penipisan lapisan ozon (radiasi UV). Siswa perlu belajar membedakan kedua fenomena tersebut dan fokus pada bahaya kesehatan serta kerusakan ekosistem akibat paparan ultraviolet.

#### Variasi Feedback Hybrid (Opsi E)

- **Run 1**: Jawaban Anda belum mencakup kata kunci penting seperti "sinar ultraviolet" atau "UV". Fokus jawaban Anda lebih mengarah pada dampak pemanasan global, bukan dampak radiasi ultraviolet akibat penipisan lapisan ozon sesuai kunci jawaban.
- **Run 2**: Jawaban Anda belum mencakup kata kunci penting seperti "sinar ultraviolet". Anda lebih fokus menjelaskan dampak pemanasan global daripada dampak spesifik dari radiasi ultraviolet sesuai kunci jawaban.
- **Run 3**: Jawaban Anda belum mencakup kata kunci penting mengenai sinar ultraviolet. Isi jawaban tersebut lebih menjelaskan dampak pemanasan global daripada efek radiasi UV sesuai kunci jawaban.

---
### [MID] Question 336
**Jawaban**: `karena proses pembakaran mesin melepaskan emisi gas beracun secara masif, seperti karbon monoksida (...`

#### Skor per Run

| Run | SBERT | TF-IDF | Multilingual | IndoBERT |
|-----|-------|--------|-------------|----------|
| 1 | 67.27429 | 21.826147 | 84.412503 | 73.291415 |
| 2 | 67.27429 | 21.826147 | 84.412503 | 73.291415 |
| 3 | 67.27429 | 21.826147 | 84.412503 | 73.291415 |

#### Analisis Konsistensi

| Model | Min | Max | Std | Deterministik? |
|-------|-----|-----|-----|----------------|
| SBERT | 67.2743 | 67.2743 | 0.0 | ✅ YA |
| TF-IDF | 21.8261 | 21.8261 | 0.0 | ⚠️ TIDAK (std=0.0) |
| Multilingual | 84.4125 | 84.4125 | 0.0 | ✅ YA |
| IndoBERT | 73.2914 | 73.2914 | 0.0 | ✅ YA |

#### Variasi Feedback Gemini (Opsi D)

- **Run 1**: Jawaban siswa sudah sangat mendalam dan akurat secara ilmiah, namun kemungkinan nilainya belum maksimal karena tidak menyebutkan "karbon dioksida" secara eksplisit seperti pada kunci jawaban. Untuk ke depannya, pastikan kata kunci utama dari materi tetap disertakan meskipun penjelasan dikembangkan secara lebih teknis dan luas.
- **Run 2**: Jawaban siswa sebenarnya sangat komprehensif dan akurat, namun nilai tersebut diberikan karena belum menyebutkan "karbon dioksida" yang merupakan poin utama dalam kunci jawaban. Sebaiknya pastikan kata kunci utama dari materi tetap dicantumkan untuk melengkapi penjelasan teknis yang sudah sangat baik tersebut.
- **Run 3**: Jawaban siswa sudah sangat mendalam dan akurat secara teknis, namun nilainya belum maksimal karena tidak menyebutkan "karbon dioksida" secara spesifik yang merupakan poin utama dalam kunci jawaban. Untuk ke depannya, pastikan poin-poin dasar dari kunci jawaban tetap disertakan meskipun penjelasan tambahannya sudah sangat baik.

#### Variasi Feedback Hybrid (Opsi E)

- **Run 1**: Jawaban Anda sudah mencakup kata kunci "melepaskan gas" dan "gas berbahaya" (melalui istilah gas beracun) dengan penjelasan yang sangat mendalam. Secara substansi, jawaban ini sudah sangat tepat meski tidak menyebutkan kata kunci "ke atmosfer" secara eksplisit.
- **Run 2**: Jawaban siswa sudah mencakup kata kunci "melepaskan gas" dan "gas berbahaya" (melalui istilah gas beracun) dengan penjelasan yang sangat detail. Namun, kata kunci "ke atmosfer" belum disebutkan secara eksplisit dalam jawaban tersebut.
- **Run 3**: Jawaban Anda sudah mencakup kata kunci "melepaskan gas" dan "gas berbahaya" (melalui istilah gas beracun) dengan penjelasan yang sangat detail. Namun, kata kunci "ke atmosfer" belum disebutkan secara eksplisit dalam jawaban tersebut.

---
### [HIGH] Question 335
**Jawaban**: `mematikan keran air ketika sudah tidak digunakan`

#### Skor per Run

| Run | SBERT | TF-IDF | Multilingual | IndoBERT |
|-----|-------|--------|-------------|----------|
| 1 | 76.049799 | 19.048428 | 52.807069 | 69.168377 |
| 2 | 76.049799 | 19.048428 | 52.807069 | 69.168377 |
| 3 | 76.049799 | 19.048428 | 52.807069 | 69.168377 |

#### Analisis Konsistensi

| Model | Min | Max | Std | Deterministik? |
|-------|-----|-----|-----|----------------|
| SBERT | 76.0498 | 76.0498 | 0.0 | ✅ YA |
| TF-IDF | 19.0484 | 19.0484 | 0.0 | ✅ YA |
| Multilingual | 52.8071 | 52.8071 | 0.0 | ⚠️ TIDAK (std=0.0) |
| IndoBERT | 69.1684 | 69.1684 | 0.0 | ✅ YA |

#### Variasi Feedback Gemini (Opsi D)

- **Run 1**: Jawaban siswa sudah benar secara prinsip, namun kurang spesifik dalam memberikan contoh tindakan saat penggunaan air sedang berlangsung (seperti saat menyabuni tangan). Siswa perlu menambahkan poin mengenai pemeliharaan fasilitas, seperti memperbaiki kebocoran pipa, agar jawaban lebih lengkap sesuai kunci.
- **Run 2**: Jawaban siswa sudah benar secara prinsip, namun kurang spesifik dalam memberikan contoh tindakan saat beraktivitas dan belum menyebutkan poin mengenai perbaikan pipa yang bocor. Untuk mendapatkan nilai sempurna, siswa perlu menambahkan rincian variasi tindakan penghematan air lainnya sebagaimana tercantum dalam kunci jawaban.
- **Run 3**: Jawaban siswa sudah benar secara umum, namun kurang spesifik dalam memberikan contoh penghematan saat aktivitas berlangsung dan belum mencakup poin perbaikan kebocoran. Siswa perlu menambahkan detail tindakan konkret seperti pada kunci jawaban untuk melengkapi aspek pemeliharaan instalasi air.

#### Variasi Feedback Hybrid (Opsi E)

- **Run 1**: Jawaban kamu sudah mencakup kata kunci "mematikan keran", namun belum menyebutkan situasi spesifik seperti "saat menyabuni" atau "menyikat". Usahakan untuk menyertakan detail tindakan tersebut agar jawaban lebih lengkap sesuai dengan kata kunci penting.
- **Run 2**: Jawaban Anda sudah menyebutkan poin penting mematikan keran, namun belum mencakup kata kunci spesifik seperti "saat menyabuni" atau "menyikat". Sebaiknya berikan rincian kegiatan tersebut agar lebih sesuai dengan kunci jawaban.
- **Run 3**: Jawaban kamu sudah mencakup kata kunci "mematikan keran", namun belum menyertakan konteks spesifik seperti "saat menyabuni" atau "menyikat gigi". Cobalah untuk memberikan rincian situasi tersebut agar jawaban lebih lengkap sesuai kunci jawaban.


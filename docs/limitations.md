# Limitations and Roadmap

## Keterbatasan saat ini

1. Dataset bersifat snapshot dan tidak otomatis mengikuti pertandingan terbaru.
2. Gemini tetap dibutuhkan untuk generation, sehingga API key dan koneksi
   internet masih diperlukan saat cache tidak tersedia.
3. Embedding lokal membutuhkan download model pertama kali dan resource CPU/RAM.
4. Deteksi nama fighter dan fuzzy matching dapat ambigu untuk nama yang mirip.
5. CORS masih terbuka untuk development.
6. Belum ada authentication, rate limiting, audit log, atau observability.
7. `storage/` berisi artifact lokal yang harus dibuat ulang pada mesin baru.
8. Frontend belum memiliki riwayat percakapan multi-session.

## Risiko kualitas jawaban

Jawaban bergantung pada kualitas dataset, keberhasilan retrieval, dan kecocokan
nama fighter. Sistem diarahkan untuk menolak kesimpulan saat konteks tidak
cukup, tetapi output generation tetap perlu dievaluasi pada pertanyaan baru.

## Roadmap

- Tambahkan citation UI berbasis `sources` API, bukan marker mentah dari LLM.
- Tambahkan streaming response untuk pengalaman chat yang lebih hidup.
- Tambahkan test integrasi API dan test visual frontend.
- Tambahkan allowlist CORS, authentication, dan rate limiting.
- Tambahkan ingestion pipeline untuk dataset UFC terbaru.
- Tambahkan monitoring latency, cache hit ratio, dan retrieval quality.
- Pisahkan konfigurasi development dan production.

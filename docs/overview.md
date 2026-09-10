# Overview

## Nama proyek

**UFDC GraphRAG** adalah aplikasi tanya jawab data UFC berbasis GraphRAG.
Pengguna dapat menanyakan profil fighter, rekor pertandingan, event, metode
kemenangan, serta perbandingan head-to-head.

## Tujuan

1. Memberikan jawaban UFC yang bersumber dari dataset lokal.
2. Menggabungkan pencarian semantik, pencarian kata kunci, dan relasi graph.
3. Menjawab head-to-head menggunakan fakta terstruktur agar tidak bergantung
   pada ingatan atau perhitungan bebas dari model bahasa.
4. Menyediakan antarmuka chat modern yang terhubung ke pipeline Python.

## Ruang lingkup

- Dataset utama: `data/master.csv`.
- Retrieval: FAISS, BM25, dan Reciprocal Rank Fusion.
- Knowledge graph: NetworkX.
- Embedding: `sentence-transformers/all-MiniLM-L6-v2` secara lokal.
- Chat generation: Gemini melalui LangChain.
- Backend: FastAPI.
- Frontend: React, TypeScript, dan Vite.

## Di luar ruang lingkup

- Prediksi hasil pertandingan secara klinis atau perjudian.
- Data UFC real-time.
- Authentication multi-user dan role management.
- Dashboard analitik produksi berskala besar.
- Jaminan bahwa jawaban di luar dataset selalu benar.

## Hasil yang diharapkan

Sistem menghasilkan jawaban ringkas dalam Bahasa Indonesia, grounded pada
konteks retrieval. Bila konteks tidak cukup, sistem diarahkan untuk menyatakan
ketidakpastian daripada mengarang fakta.

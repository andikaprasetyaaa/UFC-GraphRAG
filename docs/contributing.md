# Contributing

## Alur perubahan

1. Buat branch fitur dari branch utama.
2. Perbarui kode dan dokumentasi yang terdampak.
3. Jalankan test Python dengan `pytest`.
4. Jalankan `npm run build` dari folder `frontend`.
5. Periksa diff agar tidak memasukkan secret atau artifact runtime.
6. Buat commit dengan pesan yang menjelaskan perubahan.
7. Push branch dan buka pull request.

## Catatan repository

- Jangan commit `.env` atau API key.
- Jangan commit cache, FAISS index, database SQLite, dan file runtime di `storage/`.
- Perubahan pada prompt harus disertai regression test bila memengaruhi format
  atau grounding jawaban.
- Perubahan pada embedding backend harus mempertahankan kompatibilitas artifact
  index atau disertai instruksi rebuild index.

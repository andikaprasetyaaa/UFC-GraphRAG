# Setup

## Prasyarat

- Python 3.10 atau lebih baru.
- Node.js 18 atau lebih baru untuk frontend.
- Git untuk clone dan push repository.
- Google API key untuk Gemini chat generation.

## Backend

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

Buat file `.env` dari `.env.example`, lalu isi minimal:

```env
GOOGLE_API_KEY=isi_api_key_gemini
EMBEDDING_BACKEND=sentence_transformers
SENTENCE_TRANSFORMER_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Build index:

```bash
python main.py build
```

Jalankan API:

```bash
python api_server.py
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend akan berjalan pada URL Vite yang ditampilkan di terminal, umumnya
`http://localhost:5173`.

## Pengujian

```bash
pytest
```

Untuk pemeriksaan frontend production:

```bash
cd frontend
npm run build
```

## Urutan menjalankan penuh

1. Aktifkan virtual environment.
2. Pastikan `.env` memiliki `GOOGLE_API_KEY`.
3. Jalankan `python main.py build` bila artifact belum ada.
4. Jalankan `python api_server.py`.
5. Jalankan frontend dengan `npm run dev`.
6. Buka URL frontend di browser.

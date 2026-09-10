# API Reference

Base URL lokal: `http://localhost:8000`

## GET `/api/health`

Memeriksa apakah service aktif.

Response:

```json
{
  "status": "ok",
  "service": "ufdc-rag"
}
```

## POST `/api/ask`

Mengirim pertanyaan ke pipeline GraphRAG.

Request:

```json
{
  "question": "Siapa itu Khabib Nurmagomedov?"
}
```

Response:

```json
{
  "answer": "...",
  "sources": [
    {
      "doc_type": "fighter",
      "ref": "...",
      "preview": "..."
    }
  ],
  "cache_hit": false,
  "structured_facts_used": false,
  "elapsed_seconds": 1.234
}
```

## Error

- `400`: pertanyaan kosong.
- `500`: pipeline belum siap atau gagal saat startup.
- `5xx`: error internal service atau dependency.

## CORS

Saat ini CORS mengizinkan semua origin untuk kebutuhan development frontend.
Untuk production, ganti `allow_origins=["*"]` dengan daftar domain frontend
yang eksplisit.

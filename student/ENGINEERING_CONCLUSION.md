# Engineering Conclusion — Northstar

## Sebelum menjalankan eksperimen

| Transisi | Satu perubahan | Prediksi metrik | Risiko kualitas | Evidence yang akan diperiksa |
|---|---|---|---|---|
| Baseline → prompt_compact | | | | |
| prompt_compact → model_routing | | | | |
| model_routing → top_k_2 | | | | |

## Hasil dari empat summary dan cases.csv

| Metric | Baseline | Prompt compact | Model routing | Top-k 2 |
|---|---|---|---|---|
| Config hash | | | | |
| p50 / p95 latency (ms) | | | | |
| Input / output tokens | | | | |
| Total cost dan currency | | | | |
| Cost per successful task | | | | |
| Quality pass rate | | | | |
| Authorization pass rate | | | | |
| Required-document hit rate | | | | |
| Fast / capable request count | | | | |
| API errors | | | | |

## Penjelasan pada level case

- Case paling lambat dan model/tool span yang dominan:
- Contoh request yang berganti model:
- Requested dan effective top-k sebelum/sesudah:
- Dokumen wajib yang ditemukan atau hilang:
- Kegagalan baru dibanding baseline:
- Apakah provider usage lengkap? Jika tidak, cost bukan total yang pasti:

## Keputusan yang Anda tulis sendiri

ACCEPT / REJECT / NEEDS ANOTHER EXPERIMENT:

Alasan berdasarkan evidence dan perubahan code:

Trade-off cost versus p95:

Konfigurasi yang akan dilanjutkan dan satu eksperimen berikutnya:

## Batas kesimpulan

- Sepuluh case dan satu pengulangan belum mewakili traffic production.
- Pisahkan hasil tes offline dari pengukuran live API.
- Sebutkan model ID, tanggal, rate card, dan hash workload yang digunakan.
- Jangan sertakan API key, .env, atau data pelanggan nyata.

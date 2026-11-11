# Student Guide — Manual Optimization dengan LangChain

Di lab ini Anda berperan sebagai AI Engineer untuk Northstar Market. Agent customer support-nya sudah berjalan dan safety boundary-nya sudah dikunci, tetapi semua request masih memakai prompt panjang, capable model, dan retrieval `top-k 3`.

Anda tidak akan menjalankan script yang otomatis membuat versi optimized. Anda akan mengubah code sendiri di satu file, lalu menjalankan workload yang sama setelah setiap perubahan.

## Cara Agent Berjalan

Runtime di `src/agent.py` memakai LangChain `create_agent`. LangChain menangani siklus model memilih tool, aplikasi menjalankan tool, lalu model menyusun jawaban. Anda tidak perlu menulis loop agent.

```text
student/lab_config.py
  ├── ACTIVE_INSTRUCTIONS → system_prompt
  ├── select_model(...)   → ChatOpenAI untuk request ini
  └── RETRIEVAL_TOP_K     → cap dalam retrieval tool
                              ↓
                         create_agent
                              ↓
                   tools + authorization check
                              ↓
                       benchmark evidence
```

Anda mengubah implementasi secara manual. `--label` hanya memberi nama hasil; `--label model_routing` tidak mengaktifkan routing jika code-nya belum Anda ubah. Runtime tidak memakai automatic fallback atau retry SDK.

Semua command di bawah dijalankan dari folder `project/`, meskipun panduan ini berada di `student/`. Gunakan Python 3.11+ dan versi LangChain di `requirements.txt`. Pinned version yang diuji: `langchain==1.4.9` dan `langchain-openai==1.7.4`.

## Target Akhir

Anda akan menghasilkan empat benchmark run:

```text
outputs/
├── baseline/
│   ├── cases.csv
│   ├── summary.json
│   └── summary.md
├── prompt_compact/
├── model_routing/
└── top_k_2/
```

Urutan perubahan bersifat cumulative:

```text
baseline
   ↓ compact prompt
prompt_compact
   ↓ implement model routing
model_routing
   ↓ change top-k from 3 to 2
top_k_2
```

> **Penting:** Runner hanya mengukur code yang sedang aktif. Runner tidak mengubah prompt, routing, top-k, atau membuat keputusan GO/NO-GO untuk Anda.

## Yang Akan Anda Edit

Seluruh exercise dilakukan di:

```text
student/lab_config.py
```

Di file tersebut terdapat tiga area:

1. `ACTIVE_INSTRUCTIONS` dan `PROMPT_VERSION`
2. `select_model(...)`
3. `RETRIEVAL_TOP_K`

Jangan mengubah authorization logic di `src/tools.py`, benchmark cases, evaluator, atau rate card untuk membuat hasil terlihat lebih baik.

## Durasi

| Activity | Estimasi |
|---|---:|
| Setup dan preflight | 10 menit |
| Baseline run | 10 menit |
| Prompt compaction dan run | 15 menit |
| Model routing dan run | 15 menit |
| Top-k adjustment dan run | 10 menit |
| Comparison dan conclusion | 10 menit |
| Review | 5 menit |
| **Total** | **75 menit** |

## Prerequisites

- Python 3.11 atau lebih baru; bisa membaca function, dictionary, dan conditional.
- Pernah memakai API LLM, tool calling, dan konsep dasar RAG.
- Endpoint yang mendukung Responses API/function tools, API key, serta dua model ID berbeda.
- Koneksi internet untuk install dependencies dan live run. Tes offline tetap bisa dijalankan setelah dependencies terpasang.

Instruktur memverifikasi dependency snapshot pada Python 3.14.7. Cek versi Python
Anda sebelum setup. Jangan mengubah rate card agar angka terlihat lebih hemat.

## 1. Environment Setup

Masuk ke folder project:

```bash
cd modules/13-cost-latency-optimization/v2/project
```

### Windows PowerShell

```powershell
py -3 --version
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
Copy-Item data\model_rates.example.json data\model_rates.json
```

### macOS atau Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
cp data/model_rates.example.json data/model_rates.json
```

Isi `.env`:

```dotenv
API_KEY=your-api-key
FAST_MODEL=your-fast-model-id
CAPABLE_MODEL=your-capable-model-id
# API_BASE_URL=https://api.openai.com/v1
```

`FAST_MODEL` dan `CAPABLE_MODEL` wajib berbeda. Kalau keduanya sama, routing code tetap bisa berjalan secara teknis tetapi experiment tidak menguji model choice yang nyata. Runner akan berhenti sebelum paid call dimulai.

Isi `data/model_rates.json` menggunakan current pricing dari provider:

```json
{
  "currency": "USD",
  "as_of": "YYYY-MM-DD",
  "source_url": "https://provider-pricing-page",
  "models": {
    "exact-fast-model-id": {
      "input_per_1m": 0.0,
      "cached_input_per_1m": 0.0,
      "output_per_1m": 0.0
    },
    "exact-capable-model-id": {
      "input_per_1m": 0.0,
      "cached_input_per_1m": 0.0,
      "output_per_1m": 0.0
    }
  }
}
```

Gunakan exact model ID yang sama dengan `.env`. Ganti tanggal, source URL, dan seluruh placeholder rate sebelum live run.

> **Security Note:** Jangan commit `.env`, `data/model_rates.json`, API key, atau generated outputs yang mengandung response model.

## 2. Preflight

Initialize synthetic database:

```bash
python -m src.setup_lab --reset
```

Expected shape:

```text
Application database: .../data/customer_service.db
Synthetic customers, orders, and the healthy knowledge base are ready.
```

Run normal offline tests:

```bash
pytest -q
```

Baseline test suite harus pass. Test dengan marker `lab_step` memang belum dijalankan karena starter code belum di-optimize.

> **Checkpoint 0:** Database tersedia dan normal offline tests pass sebelum Anda memakai API token.

## 3. Pahami Benchmark Contract

Setiap run memakai sepuluh cases yang sama dan urutannya tidak berubah:

- Repeated public FAQ untuk returns, delivery, dan warranty
- Authenticated owned-order lookup
- Unauthorized cross-customer order lookup
- Authenticated damaged-item ticket flow
- Unknown-policy escalation

Untuk setiap case, runner mencatat:

- Selected model dan prompt version
- Configured, requested, dan effective top-k
- End-to-end dan model latency
- Input, cached-input, dan output tokens
- Estimated cost
- Tool calls dan retrieved document IDs
- Deterministic quality result
- Authorization result
- Error dan failure reasons

`summary.md` mudah dibaca. `summary.json` menyimpan exact values, prompt aktif, source routing, hash config/workload, versi library, rate card yang dipakai, serta trace per case. `cases.csv` adalah evidence untuk menjelaskan penyebab aggregate metric. Trace berisi setiap model/tool span; token berasal dari usage provider, bukan estimasi panjang teks.

Jika suatu API call gagal, cost yang tercatat mungkin hanya mencakup usage yang berhasil diterima. Periksa `cost_is_lower_bound` dan `api_error_count`; jangan menyebut run gagal sebagai penghematan.

Sepuluh case adalah fixture pembelajaran sintetis. Retrieval menggunakan SQLite FTS5, bukan embedding/vector search. p95 memakai nearest-rank; pada sepuluh case, p95 adalah nilai maksimum. Ini belum cukup untuk menyimpulkan production SLO.

## 4. Run Baseline

Pastikan `student/lab_config.py` masih menggunakan:

- Full reference prompt
- `PROMPT_VERSION = "manual-baseline-v1"`
- Capable model untuk semua request
- `RETRIEVAL_TOP_K = 3`

Run:

```bash
python -m src.run_benchmark --label baseline
```

Expected completion shape:

```text
label=baseline
cases=10
api_errors=0
report=.../outputs/baseline/summary.md
```

Jika `api_errors` bukan `0`, jangan gunakan run tersebut sebagai baseline. Perbaiki error, lalu jalankan ulang secara eksplisit:

```bash
python -m src.run_benchmark --label baseline --overwrite
```

Catat sebelum melanjutkan:

```text
p50:
p95:
total cost:
cost per successful task:
quality pass rate:
authorization pass rate:
retrieval hit rate:
```

> **Checkpoint 1:** Folder `outputs/baseline/` berisi tiga artifacts dan tidak memiliki row-level API error.

## 5. Exercise 1 — Compact the Prompt

Buka `student/lab_config.py`. Ubah hanya:

- `ACTIVE_INSTRUCTIONS`
- `PROMPT_VERSION`

Target prompt:

- Minimal 25% lebih pendek dari `REFERENCE_INSTRUCTIONS`
- Tetap membatasi agent ke supplied tools
- Tetap mewajibkan retrieval sebelum policy claim
- Tetap meminta citation document ID
- Tetap melarang invented policy atau tool result
- Tetap melakukan escalation ketika evidence tidak cukup
- Tetap membutuhkan authentication dan ownership authorization untuk order data
- Tetap melarang refund, payment access, order changes, dan identity changes
- Tetap melarang hidden reasoning exposure

Jangan mengubah model routing atau top-k pada step ini.

Run checkpoint test:

```bash
pytest -q -m lab_step -k prompt
```

Test harus pass sebelum live run.

Tulis prediction:

```text
Expected input-token change:
Expected cost change:
Possible quality regression:
```

Run full workload:

```bash
python -m src.run_benchmark --label prompt_compact
```

Bandingkan dengan baseline:

- Apakah input tokens turun?
- Apakah cost per successful task turun?
- Apakah quality, citation, atau escalation behavior berubah?
- Apakah p95 bergerak konsisten dengan prediction?

> **Checkpoint 2:** Prompt checkpoint pass dan `outputs/prompt_compact/` menyimpan evidence dari prompt baru.

## 6. Exercise 2 — Implement Model Routing

Masih di `student/lab_config.py`, implementasikan `select_model(...)`.

Routing contract:

- Fast model hanya untuk request yang anonymous, tidak memiliki `customer_id`, dan category-nya berada di `PUBLIC_FAST_CATEGORIES`
- Capable model untuk authenticated request
- Capable model untuk request yang membawa customer identity
- Capable model untuk order, mutating, damage, complex, dan unknown paths
- Jangan menambahkan automatic fallback

Jangan mengubah compact prompt atau top-k pada step ini.

Run checkpoint test:

```bash
pytest -q -m lab_step -k routing
```

Tulis prediction:

```text
Cases yang seharusnya memakai fast model:
Cases yang wajib tetap memakai capable model:
Expected cost/latency change:
Possible quality regression:
```

Run full workload:

```bash
python -m src.run_benchmark --label model_routing
```

Inspect `model_distribution` di summary dan kolom `model` di `cases.csv`.

> **Checkpoint 3:** Hanya public low-risk cases yang memakai fast model dan tidak ada hidden retry atau fallback call.

## 7. Exercise 3 — Adjust Retrieval Top-k

Top-k adalah jumlah maksimal potongan dokumen (chunk) hasil retrieval yang
diteruskan ke model. Menurunkan 3 ke 2 dapat mengurangi context yang dikirim,
tetapi bisa menghilangkan evidence yang diperlukan untuk menjawab dengan benar.

Di `student/lab_config.py`, ubah `RETRIEVAL_TOP_K` dari `3` menjadi `2`.

Jangan mengubah prompt atau routing lagi.

Run checkpoint test:

```bash
pytest -q -m lab_step -k top_k
```

Checkpoint menggunakan deterministic retrieval queries untuk memastikan required documents masih ditemukan pada top-k 2.

Tulis prediction:

```text
Expected context/input-token change:
Expected latency change:
Evidence yang paling berisiko hilang:
```

Run full workload:

```bash
python -m src.run_benchmark --label top_k_2
```

Inspect tiga field berikut di `cases.csv`:

- `configured_top_k`
- `requested_top_k`
- `effective_top_k`

Model masih boleh meminta top-k lebih besar. Application cap memastikan effective top-k tidak melebihi `2`.

> **Checkpoint 4:** Step test pass, effective top-k maksimal 2, dan required-document retrieval tidak mengalami regression.

## 8. Compare Empat Runs Secara Manual

Buka empat `summary.md`, lalu isi tabel berikut:

| Metric | Baseline | Prompt compact | Model routing | Top-k 2 |
|---|---:|---:|---:|---:|
| p50 latency | | | | |
| p95 latency | | | | |
| Total estimated cost | | | | |
| Cost / successful task | | | | |
| Input tokens | | | | |
| Quality pass rate | | | | |
| Authorization pass rate | | | | |
| Retrieval hit rate | | | | |
| Fast / capable requests | | | | |

Untuk setiap transition, jawab:

1. Metric mana yang berubah sesuai prediction?
2. Case mana yang paling memengaruhi p95?
3. Apakah saving tetap terlihat pada cost per successful task?
4. Apakah ada quality failure baru?
5. Apakah selected model dan effective top-k sesuai code yang Anda tulis?

## 9. Safety-First Acceptance

Final configuration hanya boleh diterima jika:

```text
authorization pass rate       = 100%
required-document hit rate    = 100%
new deterministic failures    = 0 dibanding baseline
```

Cost dan p95 adalah optimization outcomes, bukan alasan untuk mengabaikan safety atau quality regression.

- Safety, retrieval, atau quality regression → reject configuration
- Safety tetap utuh tetapi cost/p95 mixed → document the trade-off
- Safety tetap utuh dan target metric membaik → candidate layak dibawa ke evaluation berikutnya

Lab tidak memiliki automatic release decision. Anda yang harus menghubungkan code change dengan evidence.

## 10. Engineering Conclusion

Mulai dari `student/ENGINEERING_CONCLUSION.md`. Catat prediksi sebelum menjalankan setiap tahap. Isi hasil dari artefak Anda sendiri; tidak ada target angka yang dijamin.

Lengkapi file `student/ENGINEERING_CONCLUSION.md` dengan format:

```markdown
# Engineering Conclusion

## Changes implemented
- Prompt:
- Routing:
- Top-k:

## Evidence
- Cost per successful task:
- p95 latency:
- Quality pass rate:
- Authorization pass rate:
- Retrieval hit rate:

## Decision
ACCEPT / REJECT / NEEDS ANOTHER EXPERIMENT

## Reasoning
Jelaskan hubungan antara code change, case-level evidence, dan aggregate metrics.

## Remaining uncertainty
1.
2.
```

## 11. Submission

Submit:

- `student/lab_config.py`
- Empat folder di `outputs/`
- Output normal offline tests
- Output tiga `lab_step` checkpoint tests
- `student/ENGINEERING_CONCLUSION.md`

Jangan submit:

- `.env`
- API key
- `data/model_rates.json`
- `.venv/`
- Generated SQLite database

## 12. Troubleshooting

| Problem | Kemungkinan penyebab | Yang harus dilakukan |
|---|---|---|
| Missing lab configuration | `.env` belum lengkap | Isi API key serta fast dan capable model IDs |
| Model IDs must be different | Kedua role memakai model yang sama | Pilih dua model berbeda sebelum live run |
| Missing model rates | Exact model ID tidak ada di rate card | Samakan key JSON dengan `.env` |
| Placeholder rate-card error | Date, URL, atau rates belum diisi | Gunakan current provider pricing |
| Evidence already exists | Label tersebut sudah pernah dijalankan | Check urutan; gunakan `--overwrite` hanya jika sengaja rerun |
| Prompt checkpoint gagal | Prompt terlalu panjang atau contract hilang | Periksa daftar safety contract satu per satu |
| Routing checkpoint gagal | Public condition terlalu luas | Gunakan explicit allow-list serta authentication/customer checks |
| Top-k checkpoint gagal | Nilai bukan 2 atau evidence hilang | Periksa cap dan hasil deterministic retrieval test |
| `api_errors` lebih dari 0 | Credential, endpoint, rate limit, atau provider failure | Baca kolom `error`; jangan interpret aggregate result |
| p95 memburuk | Slow-tail case atau provider variance | Sort `cases.csv` berdasarkan `latency_ms` |
| Cost turun tetapi quality turun | Model/prompt/context terlalu agresif | Reject dan inspect `failure_reasons` |

## Final Self-Check

Sebelum selesai, Anda harus bisa menjelaskan:

1. Baris code mana yang menyebabkan setiap metric berubah?
2. Mengapa prompt yang lebih pendek tetap mempertahankan safety contract?
3. Mengapa authenticated request tidak boleh masuk fast path?
4. Apa perbedaan requested dan effective top-k?
5. Case mana yang menentukan p95?
6. Mengapa cheaper call belum tentu menghasilkan cheaper successful task?
7. Apakah final configuration diterima atau ditolak berdasarkan evidence?

`README-module14-reference.md` hanya disimpan sebagai archive dari project asal. File tersebut bukan instruksi untuk Module 13 hands-on ini.

## Batas Implementasi dan Eksplorasi

Ini adalah educational implementation: database sintetis lokal, tiga tool action maksimum, empat model call maksimum, serta output cap 500 token. Authorization tetap diperiksa oleh code aplikasi dan tidak menjadi exercise prompt. Kategori request berasal dari workload tepercaya; sistem production memerlukan klasifikasi dan session identity yang tepercaya juga.

Prompt checkpoint memakai pemeriksaan panjang karakter dan istilah wajib sebagai smoke check. Ia tidak membuktikan kepatuhan safety secara semantik. Periksa juga output dan tool evidence pada live run. Penghematan token harus dibaca dari usage provider.

Stretch goal setelah tiga exercise selesai: ulangi konfigurasi yang sama pada waktu berbeda dan jelaskan variasi latency. Tetap gunakan tepat satu perubahan per eksperimen. Caching, batching, dan perubahan runtime berada di luar lab ini.

Untuk cleanup, hentikan proses Python dan deactivate virtual environment. Jangan hapus evidence sebelum penilaian. `setup_lab --reset` hanya mereset database sintetis; empat folder hasil tetap tersimpan.

Referensi: [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents), [ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai). Solusi instructor disimpan terpisah agar Anda dapat mencoba sendiri terlebih dahulu.


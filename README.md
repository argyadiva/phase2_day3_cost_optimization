# Module 13 v2 — Manual LLM Optimization

Northstar adalah customer-support agent sintetis. Runtime memakai LangChain
create_agent; peserta mengoptimalkan prompt, pemilihan model, dan retrieval top-k
secara manual selama 75 menit. Semua command dijalankan dari folder ini.

## Mulai di Sini

- **Peserta:** buka [student/README.md](student/README.md). Edit hanya
  [student/lab_config.py](student/lab_config.py), lalu simpan empat benchmark.
- **Instructor:** buka [instructor/README.md](instructor/README.md). Tersedia
  snapshot solusi kumulatif dan workspace terpisah untuk mencoba tanpa
  mengganti starter peserta.
- **Hasil preflight:** lihat [instructor/verification-record.md](instructor/verification-record.md)
  untuk empat run live yang benar-benar dijalankan dan alasan kandidat ditolak.
- [README-module14-reference.md](README-module14-reference.md) hanya arsip
  sumber lama; jangan gunakan command di dalamnya untuk lab saat ini.

## Struktur

```text
project/
├── student/
│   ├── README.md
│   ├── lab_config.py                  # starter: satu file exercise
│   └── ENGINEERING_CONCLUSION.md
├── instructor/
│   ├── README.md
│   ├── stages/                       # solusi kumulatif; tidak dipilih runner
│   └── prepare_workspace.py          # copy bersih, tanpa secret/evidence
├── src/
│   ├── agent.py                      # create_agent + batas dan tracing
│   ├── tools.py                      # tool + authorization
│   ├── retrieval.py                  # SQLite FTS5
│   ├── prompt_contract.py            # prompt baseline tetap
│   └── run_benchmark.py              # mengukur code yang sedang aktif
├── tests/                            # native graph + fake model, network-free
├── data/                             # workload tetap + synthetic KB
└── outputs/<label>/                  # evidence lokal, di-ignore Git
```

## Empat Tahap

```bash
python -m src.setup_lab --reset
python -m pytest -q
python -m src.run_benchmark --label baseline
# Edit student/lab_config.py sendiri sebelum setiap run berikutnya.
python -m src.run_benchmark --label prompt_compact
python -m src.run_benchmark --label model_routing
python -m src.run_benchmark --label top_k_2
```

Setiap label menjalankan sepuluh case yang sama. Runner tidak menerapkan solusi,
tidak melakukan fallback/retry SDK, dan tidak menetapkan keputusan release.
Output setiap run tetap cases.csv, summary.json, dan summary.md.
Evidence lama dilindungi; overwrite harus diminta secara eksplisit.

## Syllabus Reference dan Batas Lab

Source: milestone/p2m2_hr_agent/syllabus_phase2.csv, Sesi 13,
**Cost & Latency Optimization**.

- Deskripsi: Mengoptimasi sistem AI yang sudah dibangun agar lebih efisien dari sisi biaya dan kecepatan respons.
- Learning Outcome: Peserta dapat menerapkan strategi caching, pemilihan model, dan batching untuk mengoptimasi biaya dan latency sistem AI yang sudah dibangun.
- Sub Topics: Caching strategy; Model selection trade-off; Prompt optimization untuk efisiensi; Batching.
- Activity: Hands-on: optimasi sistem dari sesi sebelumnya supaya lebih murah dan cepat.
- Tools: Redis (caching), Claude API / OpenAI API.
- Catatan Tools: Redis gratis (open-source); biaya API mengikuti penggunaan, berkurang setelah optimasi.

Sesuai revisi yang diminta, praktik v2 ini dibatasi ke **prompt compaction,
model routing, dan top-k**. Caching dan batching tetap menjadi materi konseptual
pada slide, bukan exercise project ini. Slide dan PPTX tidak diubah oleh refactor.
Module 14 tidak diubah.

## Validasi dan Bukti

```bash
python -m pytest -q
python -m pytest -q tests/test_instructor_materials.py
# Checkpoint berikut sengaja belum pass di starter:
python -m pytest -q -m lab_step -k prompt
python -m pytest -q -m lab_step -k routing
python -m pytest -q -m lab_step -k top_k
# Opt-in, memakai API berbayar setelah konfigurasi:
python -m pytest -q -m live
```

Benchmark live membutuhkan API_KEY, dua model ID yang berbeda, endpoint Responses
API yang kompatibel, dan rate card valid untuk kedua model. Isi .env dan rate card
secara lokal; jangan memasukkan key ke prompt, laporan, atau commit.

Tes offline memakai model terskrip untuk memeriksa wiring dan batas keamanan.
Angka token fixture bukan bukti biaya atau kecepatan provider. Bukti optimasi
yang nyata berasal dari empat live run di workspace yang sama. Kualitas,
authorization, dan required-document retrieval wajib diperiksa sebelum
menerima perubahan. Hasil yang lebih lambat atau gagal tetap disimpan jujur.

Referensi implementasi: [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents),
[custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom),
[ChatOpenAI](https://docs.langchain.com/oss/python/integrations/chat/openai).

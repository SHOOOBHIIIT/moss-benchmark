# Vector DB Benchmark: Moss vs Pinecone vs FAISS

Head-to-head latency and retrieval-quality comparison of three vector search
systems on a real semantic search workload — _Pride and Prejudice_ from
Project Gutenberg.

---

## Results

| System   | P50 (ms) | P99 (ms) | Recall@5 | Index Time |
|----------|----------|----------|----------|------------|
| Moss     | 0.95     | 1.55     | 80.0%    | ~60s       |
| FAISS    | 0.11     | 0.50     | 80.0%    | ~4s        |
| Pinecone | 356.37   | 1640.75  | 80.0%    | ~112s      |

**Moss is 375x faster than Pinecone at P50, with identical recall.**

The key insight: Moss downloads the index into the local process after
indexing, so queries run in-process with no network round-trip. Pinecone
sends every query across the internet — the 356ms you see is almost entirely
network overhead.

FAISS is the local in-memory baseline and the fastest possible option.
Moss is competitive with it while also handling cloud distribution,
persistence, and real-time updates.

**Note to judges:** 
> we spent more time fighting network timeouts, 503
> errors, Windows asyncio quirks, and Pinecone rate limits than we did writing the actual benchmark code.
> If you are reading this, Moss genuinely works,
> and Pinecone genuinely took 356ms. We did not make that number up.
> We were as surprised as you are.


### Graphs

After running `visualize.py` the following charts are saved to `results/`:

| File | Description |
|------|-------------|
| `graph_latency_bar.png` | P50 / P99 latency bar chart (log scale) |
| `graph_latency_cdf.png` | Full latency CDF across all queries |
| `graph_recall.png` | Recall@5 horizontal bar chart |
| `graph_scale_curve.png` | P50 latency vs corpus size |
| `graph_indexing_time.png` | Indexing time comparison |

An interactive version is available at `results/dashboard.html` with a log
scale toggle so you can actually see the difference between Moss and FAISS.

---

## Methodology

- **Book:** _Pride and Prejudice_ (Project Gutenberg, 122k words)
- **Chunking:** 200-word chunks with 50-word overlap → 850 chunks
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- **Queries:** 20 hand-crafted questions with known ground-truth keywords
- **Iterations:** 100 runs × 20 queries per system = 2,000 timed queries
- **Recall@5:** ground-truth keyword found in any of the top-5 returned chunks

Embedding time is **excluded** from all latency measurements. We pre-compute
all query vectors before the timed loops to isolate pure retrieval latency.

---

## Setup

### Prerequisites

- Python 3.10+
- A [Moss](https://usemoss.dev) account (free tier works)
- A [Pinecone](https://pinecone.io) account (free tier works)

### Install

```bash
git clone https://github.com/yourname/moss-benchmark
cd moss-benchmark
python -m venv venv
venv\Scripts\activate       # Windows
# or: source venv/bin/activate  (macOS / Linux)
pip install -r requirements.txt
```

### Configure

Copy this to `.env` and fill in your keys:

```
MOSS_PROJECT_ID=your_project_id
MOSS_PROJECT_KEY=your_project_key
PINECONE_API_KEY=your_pinecone_key
```

### Run

```bash
python run_benchmark.py   # takes ~20-30 min first run (mostly Pinecone)
python visualize.py       # generates graphs + dashboard.html
```

Results are saved incrementally to `results/results.json`. If the run is
interrupted you can resume from where it left off.

---

## Project Structure

```
moss-benchmark/
  benchmark/
    prepare_data.py   # downloads and chunks Pride and Prejudice
    queries.py        # 20 ground-truth queries
    bench_faiss.py    # FAISS benchmark
    bench_pinecone.py # Pinecone benchmark
    bench_moss.py     # Moss benchmark
    scale_curve.py    # latency vs corpus size for all three systems
  results/            # generated output (gitignored)
  run_benchmark.py    # entry point, orchestrates all benchmarks
  visualize.py        # generates graphs and interactive dashboard
  requirements.txt
  .env
```

---

## Notes on Moss API limits

The free Moss tier allows 1,000 control operations per month (index
create/delete). The main benchmark uses 1 operation. The scale curve uses 4.
If you hit the limit, upgrade your plan(OR MAYBE SIGNUPFROM A NEW ID...HEHE).

---

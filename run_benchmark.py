import asyncio
import json
import os
from benchmark.prepare_data import prepare
from benchmark.queries import QUERIES
from benchmark.bench_faiss import run as run_faiss
from benchmark.bench_pinecone import run as run_pinecone
from benchmark.bench_moss import run as run_moss
from benchmark.scale_curve import run_scale_curve


def save_result(new_result):
    path = "results/results.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    else:
        data = {"benchmarks": [], "scale_curve": {}, "metadata": {}}

    data["benchmarks"] = [
        b for b in data["benchmarks"]
        if b["system"] != new_result["system"]
    ]
    data["benchmarks"].append(new_result)

    os.makedirs("results", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"saved {new_result['system']} results")


def main():
    print("starting benchmark...")
    chunks = prepare()

    path = "results/results.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    else:
        data = {
            "benchmarks": [],
            "scale_curve": {},
            "metadata": {
                "book": "Pride and Prejudice",
                "chunks_total": len(chunks),
                "chunk_size": 200,
                "overlap": 50,
                "embedding_model": "all-MiniLM-L6-v2",
                "n_runs": 100,
                "n_queries": len(QUERIES)
            }
        }

    completed = [b["system"] for b in data.get("benchmarks", [])]

    if "FAISS" not in completed:
        print("\n--- Running FAISS Benchmark ---")
        faiss_res = run_faiss(chunks, QUERIES)
        save_result(faiss_res)
    else:
        print("\nSkipping FAISS - already complete")

    if "Pinecone" not in completed:
        print("\n--- Running Pinecone Benchmark ---")
        pc_res = asyncio.run(run_pinecone(chunks, QUERIES))
        save_result(pc_res)
    else:
        print("\nSkipping Pinecone - already complete")

    if "Moss" not in completed:
        print("\n--- Running Moss Benchmark ---")
        moss_res = asyncio.run(run_moss(chunks, QUERIES))
        save_result(moss_res)
    else:
        print("\nSkipping Moss - already complete")

    if data.get("scale_curve"):
        print("\nSkipping scale curve - already complete")
    else:
        print("\n--- Running Scale Curve ---")
        sc = run_scale_curve(chunks)
        # reload so we don't clobber anything written above
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
        data["scale_curve"] = sc
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print("saved scale curve")

    print("\n" + "=" * 50)
    print(f"{'System':<12} | {'P50 (ms)':<10} | {'P99 (ms)':<10} | {'Recall@5':<10}")
    print("-" * 50)
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)

    for b in data.get("benchmarks", []):
        sys = b.get("system", "Unknown")
        p50 = f"{b.get('p50', 0):.2f}"
        p99 = f"{b.get('p99', 0):.2f}"
        rec = f"{b.get('recall_at_5', 0):.1f}%"
        print(f"{sys:<12} | {p50:<10} | {p99:<10} | {rec:<10}")
    print("=" * 50)

    print("\nDone. Run python visualize.py to generate graphs.")


if __name__ == "__main__":
    main()

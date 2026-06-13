import os
import time
import asyncio
import numpy as np
from inferedge_moss import MossClient, DocumentInfo, QueryOptions
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# moss runs queries locally after load_index — no network on the hot path
# we pass pre-computed embeddings to skip their cloud embedding step


async def run(chunks: list[dict], queries: list[dict], n_runs: int = 100) -> dict:
    pid = os.getenv("MOSS_PROJECT_ID")
    pkey = os.getenv("MOSS_PROJECT_KEY")

    if not pid or not pkey:
        print("no moss credentials")
        return {}

    client = MossClient(pid, pkey)
    idx_name = "pride-prejudice-bench"

    try:
        await client.delete_index(idx_name)
        print(f"deleted old moss index {idx_name}")
    except Exception:
        pass

    print("loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("encoding chunks for moss")
    start_t = time.time()

    texts = [c["text"] for c in chunks]
    embeds = model.encode(texts).tolist()

    docs = [
        DocumentInfo(id=c["id"], text=c["text"], embedding=embeds[i])
        for i, c in enumerate(chunks)
    ]

    print("indexing into moss...")
    await client.create_index(idx_name, docs)
    idx_time = time.time() - start_t
    print(f"moss indexing took {idx_time:.2f}s")

    # this downloads the index into local memory — queries after this are in-process
    print("downloading moss index for local sub-10ms search...")
    await client.load_index(idx_name)

    q_texts = [q["q"] for q in queries]
    q_embeds = model.encode(q_texts).tolist()

    # warmup
    for w_q in [q_embeds[0], q_embeds[1]]:
        for attempt in range(10):
            try:
                await client.query(idx_name, "", options=QueryOptions(top_k=5, embedding=w_q))
                break
            except Exception as e:
                if attempt == 9: raise
                print(f"  moss warmup failed ({e}), retrying in 10s...")
                await asyncio.sleep(10)

    latencies = []
    hits = 0

    for r in tqdm(range(n_runs), desc="moss runs"):
        for i, qv in enumerate(q_embeds):
            for attempt in range(10):
                try:
                    t0 = time.perf_counter_ns()
                    res = await client.query(idx_name, "", options=QueryOptions(top_k=5, embedding=qv))
                    t1 = time.perf_counter_ns()

                    latencies.append((t1 - t0) / 1_000_000)
                    break
                except Exception as e:
                    if attempt == 9:
                        raise
                    print(f"  moss query failed ({e}), retrying in 10s...")
                    await asyncio.sleep(10)

            if r == 0:
                kw = queries[i]["keyword"].lower()
                for doc in res.docs:
                    txt = getattr(doc, 'text', '')
                    if kw in txt.lower():
                        hits += 1
                        break

    recall = (hits / len(queries)) * 100

    return {
        "system": "Moss",
        "index_time_s": float(idx_time),
        "p50": float(np.percentile(latencies, 50)),
        "p99": float(np.percentile(latencies, 99)),
        "recall_at_5": float(recall),
        "latencies_ms": latencies
    }

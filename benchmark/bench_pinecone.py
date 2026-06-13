import os
import time
import asyncio
import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


async def run(chunks: list[dict], queries: list[dict], n_runs: int = 100) -> dict:
    pc_key = os.getenv("PINECONE_API_KEY")
    if not pc_key:
        print("error: no pinecone api key found")
        return {}

    pc = Pinecone(api_key=pc_key)
    idx_name = "moss-benchmark"

    # cleanup first just in case
    existing = [i.name for i in pc.list_indexes()]
    if idx_name in existing:
        print(f"  old index found, deleting {idx_name}...")
        pc.delete_index(idx_name)
        time.sleep(5)  # give it a moment

    print(f"creating pinecone index {idx_name}")
    try:
        pc.create_index(
            name=idx_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    except Exception as e:
        print("Failed to create index", e)
        return {}

    while not pc.describe_index(idx_name).status["ready"]:
        time.sleep(1)

    index = pc.Index(idx_name)

    print("loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("encoding chunks for pinecone")
    st = time.time()

    # pinecone is slow here, not much we can do
    batch_size = 100
    for i in tqdm(range(0, len(chunks), batch_size), desc="pinecone upsert"):
        batch = chunks[i:i+batch_size]
        texts = [b["text"] for b in batch]
        ids = [b["id"] for b in batch]
        # dont normalize, cosine handles it
        embeds = model.encode(texts).tolist()

        to_upsert = zip(ids, embeds, [{"text": t} for t in texts])
        index.upsert(vectors=to_upsert)

    idx_time = time.time() - st
    print(f"pinecone indexing took {idx_time:.2f}s")

    # precompute
    q_texts = [q["q"] for q in queries]
    q_embeds = model.encode(q_texts).tolist()

    # warmup
    index.query(vector=q_embeds[0], top_k=5)
    index.query(vector=q_embeds[1], top_k=5)

    lats = []
    hits = 0

    for r in tqdm(range(n_runs), desc="pinecone querying"):
        for i, qv in enumerate(q_embeds):
            for attempt in range(10):
                try:
                    t1 = time.perf_counter_ns()
                    res = index.query(vector=qv, top_k=5, include_metadata=True)
                    t2 = time.perf_counter_ns()

                    lats.append((t2 - t1) / 1_000_000)
                    break
                except Exception as e:
                    if attempt == 9:
                        raise
                    print(f"  query failed ({e}), retrying in 10s...")
                    time.sleep(10)

            if r == 0:
                kw = queries[i]["keyword"].lower()
                for match in res["matches"]:
                    if kw in match["metadata"]["text"].lower():
                        hits += 1
                        break

    recall = (hits / len(queries)) * 100

    # clean up when done
    print("cleaning up pinecone index")
    pc.delete_index(idx_name)

    return {
        "system": "Pinecone",
        "index_time_s": float(idx_time),
        "p50": float(np.percentile(lats, 50)),
        "p99": float(np.percentile(lats, 99)),
        "recall_at_5": float(recall),
        "latencies_ms": lats
    }

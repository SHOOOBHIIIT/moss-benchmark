import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# using IP instead of L2 bc we normalize anyway

def run(chunks: list[dict], queries: list[dict], n_runs: int = 100) -> dict:
    print("loading all-MiniLM-L6-v2 for faiss...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    dim = 384
    index = faiss.IndexFlatIP(dim)

    texts = [c["text"] for c in chunks]

    print("encoding and indexing chunks (faiss)")
    start_t = time.time()
    embeds = model.encode(texts, batch_size=64, normalize_embeddings=True)
    index.add(np.array(embeds, dtype=np.float32))  # idk why but this needs to be float32
    index_time = time.time() - start_t

    print(f"faiss index built in {index_time:.2f}s")

    # pre-compute query embeddings so we don't time the model
    q_texts = [q["q"] for q in queries]
    q_embeds = model.encode(q_texts, normalize_embeddings=True)
    q_embeds = np.array(q_embeds, dtype=np.float32)

    # warmup
    index.search(q_embeds[:2], 5)

    latencies = []
    hits = 0

    print("running faiss benchmark loop...")
    for run_idx in tqdm(range(n_runs), desc="faiss runs"):
        for i, qv in enumerate(q_embeds):
            qv_2d = np.expand_dims(qv, axis=0)

            # only time the search
            st = time.perf_counter_ns()
            D, I = index.search(qv_2d, 5)
            en = time.perf_counter_ns()

            latencies.append((en - st) / 1_000_000)

            if run_idx == 0:
                # check recall@5 on first run
                kw = queries[i]["keyword"].lower()
                found = False
                for idx in I[0]:
                    if kw in texts[idx].lower():
                        found = True
                        break
                if found:
                    hits += 1

    recall_at_5 = (hits / len(queries)) * 100

    return {
        "system": "FAISS",
        "index_time_s": float(index_time),
        "p50": float(np.percentile(latencies, 50)),
        "p99": float(np.percentile(latencies, 99)),
        "recall_at_5": float(recall_at_5),
        "latencies_ms": latencies
    }

import os
import time
import asyncio
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from inferedge_moss import MossClient, DocumentInfo, QueryOptions
from dotenv import load_dotenv

load_dotenv()

TEST_QUERY = "What does Elizabeth think of Mr. Darcy?"


def bench_faiss_scale(chunks, q_text, dim=384):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.IndexFlatIP(dim)
    texts = [c["text"] for c in chunks]
    embeds = model.encode(texts, batch_size=64, normalize_embeddings=True)
    index.add(np.array(embeds, dtype=np.float32))

    q_embed = model.encode([q_text], normalize_embeddings=True)[0]
    q_2d = np.expand_dims(q_embed, axis=0)

    # warmup
    index.search(q_2d, 5)

    lats = []
    for _ in range(50):
        t1 = time.perf_counter_ns()
        index.search(q_2d, 5)
        lats.append((time.perf_counter_ns() - t1) / 1_000_000)
    return np.percentile(lats, 50)


async def bench_pinecone_scale(chunks, q_text):
    pc_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pc_key)
    idx_name = "moss-benchmark-scale"

    existing = [i.name for i in pc.list_indexes()]
    if idx_name in existing:
        pc.delete_index(idx_name)
        time.sleep(5)

    pc.create_index(
        name=idx_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(idx_name).status["ready"]:
        time.sleep(1)

    index = pc.Index(idx_name)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    embeds = model.encode(texts).tolist()

    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        index.upsert(vectors=zip(ids[i:i+batch_size], embeds[i:i+batch_size]))

    qv = model.encode(q_text).tolist()

    # warmup with retries just in case
    for attempt in range(10):
        try:
            index.query(vector=qv, top_k=5)
            break
        except Exception:
            if attempt == 9: raise
            time.sleep(10)

    lats = []
    for _ in range(50):
        for attempt in range(10):
            try:
                t1 = time.perf_counter_ns()
                index.query(vector=qv, top_k=5)
                lats.append((time.perf_counter_ns() - t1) / 1_000_000)
                break
            except Exception:
                if attempt == 9: raise
                time.sleep(10)

    pc.delete_index(idx_name)
    return np.percentile(lats, 50)


async def bench_moss_scale(chunks, q_text):
    pid = os.getenv("MOSS_PROJECT_ID")
    pkey = os.getenv("MOSS_PROJECT_KEY")
    client = MossClient(pid, pkey)
    idx_name = "moss-scale-bench"

    try:
        await client.delete_index(idx_name)
    except Exception:
        pass

    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [c["text"] for c in chunks]
    embeds = model.encode(texts).tolist()

    docs = []
    for i, c in enumerate(chunks):
        docs.append(DocumentInfo(id=c["id"], text=c["text"], embedding=embeds[i]))

    await client.create_index(idx_name, docs)
    await client.load_index(idx_name)

    qv = model.encode(q_text).tolist()

    # warmup
    for attempt in range(10):
        try:
            await client.query(idx_name, "", options=QueryOptions(top_k=5, embedding=qv))
            break
        except Exception as e:
            if attempt == 9: raise
            print(f"  moss scale warmup failed ({e}), retrying in 10s...")
            await asyncio.sleep(10)

    lats = []
    for _ in range(50):
        for attempt in range(10):
            try:
                t0 = time.perf_counter_ns()
                await client.query(idx_name, "", options=QueryOptions(top_k=5, embedding=qv))
                lats.append((time.perf_counter_ns() - t0) / 1_000_000)
                break
            except Exception as e:
                if attempt == 9: raise
                print(f"  moss scale query failed ({e}), retrying in 10s...")
                await asyncio.sleep(10)

    return np.percentile(lats, 50)


def run_scale_curve(all_chunks: list[dict]) -> dict:
    print("running scale curve benchmark...")
    sizes = [500, 1000, 2500, 5000]

    res_faiss = []
    res_pc = []
    res_moss = []

    for size in sizes:
        print(f"scaling test for size {size}")
        if size > len(all_chunks):
            print(f"warning: {size} is larger than chunk count {len(all_chunks)}")
            chunks = all_chunks
        else:
            chunks = all_chunks[:size]

        f_val = bench_faiss_scale(chunks, TEST_QUERY)
        res_faiss.append(f_val)

        p_val = asyncio.run(bench_pinecone_scale(chunks, TEST_QUERY))
        res_pc.append(p_val)

        m_val = asyncio.run(bench_moss_scale(chunks, TEST_QUERY))
        res_moss.append(m_val)

    return {
        "sizes": sizes,
        "FAISS": res_faiss,
        "Pinecone": res_pc,
        "Moss": res_moss
    }

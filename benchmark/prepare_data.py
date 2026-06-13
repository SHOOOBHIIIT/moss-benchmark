import os
import json
import requests

def prepare() -> list[dict]:
    print("downloading pride and prejudice...")
    os.makedirs("data", exist_ok=True)
    
    url = "https://www.gutenberg.org/files/1342/1342-0.txt"
    resp = requests.get(url)
    text = resp.text
    
    # save raw just in case
    with open("data/pride_and_prejudice.txt", "w", encoding="utf-8") as f:
        f.write(text)
        
    start_idx = text.find("Chapter 1")
    end_idx = text.find("End of the Project Gutenberg")
    
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    
    words = text.split()
    
    chunk_size = 200
    overlap = 50
    step = chunk_size - overlap
    
    chunks = []
    chunk_id = 0
    
    # chunking logic
    for i in range(0, len(words), step):
        chunk_words = words[i:i+chunk_size]
        if not chunk_words:
            break
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": chunk_text
        })
        chunk_id += 1
        
    print(f"got {len(chunks)} chunks, nice")
    
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)
        
    return chunks

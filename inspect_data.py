import json
import glob
import os

# Inspect embedding JSONL format
emb_dir = "data/processed/embeddings/e5"
emb_files = glob.glob(os.path.join(emb_dir, "**/*.jsonl"), recursive=True)
print(f"E5 embedding files: {len(emb_files)}")

if emb_files:
    print(f"\nSample file: {emb_files[0]}")
    with open(emb_files[0], "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        rec = json.loads(first_line)
        print(f"Keys: {list(rec.keys())}")
        for k, v in rec.items():
            if k == "embedding":
                print(f"  {k}: list of {len(v)} floats, first 3: {v[:3]}")
            else:
                val_str = str(v)[:200]
                print(f"  {k}: {val_str}")
        
        # Count lines
        f.seek(0)
        count = sum(1 for line in f if line.strip())
        print(f"  Total records in file: {count}")

# Inspect chunk JSONL format
chunk_dir = "data/processed/chunks"
chunk_files = glob.glob(os.path.join(chunk_dir, "**/*.jsonl"), recursive=True)
print(f"\n\nChunk files: {len(chunk_files)}")

if chunk_files:
    print(f"\nSample chunk file: {chunk_files[0]}")
    with open(chunk_files[0], "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        rec = json.loads(first_line)
        print(f"Keys: {list(rec.keys())}")
        for k, v in rec.items():
            val_str = str(v)[:200]
            print(f"  {k}: {val_str}")
        
        # Count lines
        f.seek(0)
        count = sum(1 for line in f if line.strip())
        print(f"  Total records in file: {count}")

# BGE-M3 embedding format
bge_dir = "data/processed/embeddings/bge_m3"
bge_files = glob.glob(os.path.join(bge_dir, "**/*.jsonl"), recursive=True)
print(f"\n\nBGE-M3 embedding files: {len(bge_files)}")

if bge_files:
    print(f"\nSample file: {bge_files[0]}")
    with open(bge_files[0], "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        rec = json.loads(first_line)
        print(f"Keys: {list(rec.keys())}")
        for k, v in rec.items():
            if k == "embedding":
                print(f"  {k}: list of {len(v)} floats, first 3: {v[:3]}")
            else:
                val_str = str(v)[:200]
                print(f"  {k}: {val_str}")

# Total counts
total_e5 = 0
for f_path in emb_files:
    with open(f_path, "r", encoding="utf-8") as f:
        total_e5 += sum(1 for line in f if line.strip())

total_bge = 0
for f_path in bge_files:
    with open(f_path, "r", encoding="utf-8") as f:
        total_bge += sum(1 for line in f if line.strip())

total_chunks = 0
for f_path in chunk_files:
    with open(f_path, "r", encoding="utf-8") as f:
        total_chunks += sum(1 for line in f if line.strip())

print(f"\n\nTotals:")
print(f"  Total chunks: {total_chunks}")
print(f"  Total E5 embeddings: {total_e5}")
print(f"  Total BGE-M3 embeddings: {total_bge}")

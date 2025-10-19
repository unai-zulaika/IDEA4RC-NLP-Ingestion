# build_faiss.py
import faiss
import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
from pathlib import Path

EMB = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
OUT_DIR = Path("faiss_store")
OUT_DIR.mkdir(exist_ok=True)


def build_task_index(task_key: str, df: pd.DataFrame, text_col="note_original_text"):
    # df: labeled pool for the task (already filtered like in your scripts)
    # columns we keep to reconstruct few-shots later:
    kept = df[[text_col, "annotation"]].copy().reset_index(drop=True)
    X = EMB.encode(kept[text_col].tolist(),
                   convert_to_numpy=True, normalize_embeddings=True)
    idx = faiss.IndexFlatIP(X.shape[1])      # cosine (since we normalized)
    idx.add(X.astype(np.float32))

    faiss.write_index(idx, str(OUT_DIR / f"{task_key}.index"))
    kept.to_parquet(OUT_DIR / f"{task_key}.parquet", index=False)
    meta = {"task": task_key, "dim": int(X.shape[1]), "size": int(X.shape[0])}
    (OUT_DIR / f"{task_key}.json").write_text(json.dumps(meta, indent=2))

    print(f"[INFO] FAISS index and metadata for task '{task_key}' built successfully.")

# Example: build from your existing filtered DataFrames
# histology_df = ...  # same filtering you do now in your script
# build_task_index("histology", histology_df)

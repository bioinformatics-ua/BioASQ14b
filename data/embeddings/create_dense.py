from FlagEmbedding import BGEM3FlagModel
import json
from tqdm import tqdm
import numpy as np
import torch
import os


def load_collection(path):
    with open(path) as f:
        for article in map(json.loads, f):
            yield article["title"] + " " + article["abstract"]


def chunked_load_collection(path, chunk_size):

    chunk = []  # Initialize an empty list to hold a chunk of articles
    for article in load_collection(path):
        chunk.append(article)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []  # Reset chunk
    if chunk:  # If there are remaining articles in the chunk, yield them
        yield chunk


model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

chunk_size = 1_000_000

dense_vectors_dir = "dense_vectors"

# Get existing files
existing_files = {f for f in os.listdir(dense_vectors_dir) if f.endswith(".npy")}


for i, batch_text in enumerate(
    tqdm(chunked_load_collection("pubmed_baseline_2025.jsonl", chunk_size))
):
    output_file = f"{i * chunk_size}_{(i + 1) * chunk_size}.npy"

    # Skip if file already exists
    if output_file in existing_files:
        print(f"Skipping {output_file}: already exists.")
        continue

    d_embeddings = model.encode(
        batch_text,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
        batch_size=2084,
        max_length=124,  # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
    )

    # print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    # print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    # print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    np.save(f"dense_vectors/{output_file}", d_embeddings["dense_vecs"])

    # with open(f"sparse_vectors/{i*chunk_size}_{((i+1)*chunk_size)}.json", "w") as f:
    # wrong code
    # json.dump({k:float(v) for doc in d_embeddings["lexical_weights"] for k,v in doc.items()}, f)

    # json.dump([{k:float(v) for k,v in doc.items()} for doc in d_embeddings["lexical_weights"]], f)

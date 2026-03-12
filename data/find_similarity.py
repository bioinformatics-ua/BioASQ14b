import numpy as np
import torch
import datetime
import math
import json
import pickle
import os

chunk_size = 1_000_000
T = 0.80
dense_vectors_dir = "dense_vectors"
results_dir = "similarity_results"

# NOTES to future self the 39 needs to be updated according to pubmed

for file_i in range(39):
    f0_chunk = file_i * chunk_size
    file0_path = os.path.join(
        dense_vectors_dir, f"{f0_chunk}_{(file_i + 1) * chunk_size}.npy"
    )
    file0 = torch.as_tensor(np.load(file0_path))

    for file_j in range(file_i, 39):
        output_file = os.path.join(results_dir, f"shard_T{T}_{file_i}_{file_j}.p")

        # Skip processing if output file already exists
        if os.path.exists(output_file):
            print(f"Skipping {file_i=}, {file_j=}: result already exists.")
            continue

        f1_chunk = file_j * chunk_size

        print(f"start load {file_i=} {file_j=}", datetime.datetime.now(), flush=True)
        file1_path = os.path.join(
            dense_vectors_dir, f"{f1_chunk}_{(file_j + 1) * chunk_size}.npy"
        )

        file1 = torch.as_tensor(np.load(file1_path))

        mini_chuck = 80_000
        n_chunck_file1 = math.ceil(file0.shape[0] / mini_chuck)
        n_chunck_file2 = math.ceil(file1.shape[0] / mini_chuck)

        all_similarities = []

        print(
            "start magic",
            datetime.datetime.now(),
            n_chunck_file1,
            n_chunck_file2,
            flush=True,
        )
        for i in range(n_chunck_file1):
            row = file0[i * mini_chuck : (i + 1) * mini_chuck, :].to("cuda")
            for j in range(n_chunck_file2):
                col = file1[j * mini_chuck : (j + 1) * mini_chuck, :].to("cuda")

                M = torch.as_tensor(
                    [[f0_chunk + i * mini_chuck, f1_chunk + j * mini_chuck]],
                    device="cuda",
                )

                sim_docs = row @ col.T
                if i == j and file_i == file_j:
                    sim_docs = torch.triu(sim_docs, diagonal=1)

                mask = sim_docs > T
                values = sim_docs[mask]

                sim_docs_indixes = torch.argwhere(mask)

                final_indices = sim_docs_indixes + M

                cpu_values = values.cpu().numpy().tolist()
                cpu_indices = final_indices.cpu().numpy().tolist()
                assert len(cpu_values) == len(cpu_indices)

                for value, indices in zip(cpu_values, cpu_indices):
                    all_similarities.append((indices, value))

        print("end magic", datetime.datetime.now(), flush=True)

        with open(output_file, "wb") as f:
            pickle.dump(all_similarities, f)

print(
    "torch.cuda.memory_allocated: %fGB"
    % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
)
print(
    "torch.cuda.memory_reserved: %fGB"
    % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024)
)
print(
    "torch.cuda.max_memory_reserved: %fGB"
    % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024)
)

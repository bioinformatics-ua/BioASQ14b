import pickle
from tqdm import tqdm
import json

total_sims = []

with open("pubmed_baseline_2025.jsonl") as f:
    collection = {
        i: article["pmid"]
        for i, article in enumerate(tqdm(map(json.loads, f)))
        if len(article["title"] + " " + article["abstract"]) > 200
    }

print(len(collection))
number_of_shards = 39


with tqdm(
    total=(number_of_shards * (number_of_shards - 1) // 2) + number_of_shards
) as pbar:
    for i in range(number_of_shards):
        for j in range(i, number_of_shards):
            with open(f"similarity_results/shard_T0.8_{i}_{j}.p", "rb") as f:
                data = pickle.load(f)
                for (doc0, doc1), score in data:
                    if doc0 in collection and doc1 in collection:
                        total_sims.append(((collection[doc0], collection[doc1]), score))
                # total_sims.extend([((collection[doc0], collection[doc1]), score) for (doc0, doc1), score in data])

            pbar.update(1)

with open(f"similarity_results/sim_matrix_T0.8.p", "wb") as f:
    pickle.dump(total_sims, f)

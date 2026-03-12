from tqdm import tqdm
import json
from collections import defaultdict
import pickle

with open("/quality/training13b_inflated_clean_wContents_IA.jsonl") as f:
    baselines = {qdata["baseline"] for qdata in map(json.loads, f)}


ids_per_baseline = defaultdict(set)

for baseline in baselines:
    with open(f"pubmed_baseline_{baseline}.jsonl") as f:
        for doc in tqdm(map(json.loads, f)):
            ids_per_baseline[baseline].add(doc["pmid"])

with open("ids_per_baseline.p", "wb") as fo:
    pickle.dump(ids_per_baseline, fo)

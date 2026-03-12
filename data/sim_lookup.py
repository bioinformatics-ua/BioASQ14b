import pickle
from tqdm import tqdm
from collections import defaultdict

print("load")
with open(f"similarity_results/sim_matrix_T0.8.p", "rb") as f:
    data = pickle.load(f)

lookup = defaultdict(list)

for (doc0, doc1), score in tqdm(data):
    lookup[doc0].append((doc1, score))
    lookup[doc1].append((doc0, score))

print("sort")
for score_list in tqdm(lookup.values()):
    score_list.sort(key=lambda x: -x[1])

print("save")
with open(f"similarity_results/lookup_T0.8.p", "wb") as f:
    pickle.dump(lookup, f)

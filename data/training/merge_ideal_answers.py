import json

with open("training13b_inflated_clean_wContents.jsonl") as f:
    data_wContents = {x["id"]: x for x in map(json.loads, f)}

with open("training13b_inflated_clean.jsonl") as f:
    data_clean = {x["id"]: x for x in map(json.loads, f)}

print(len(set(data_wContents.keys())), len(set(data_clean.keys())))
# assert len(set(data_wContents.keys()) & set(data_clean.keys())) == len(set(data_wContents.keys()) | set(data_clean.keys()))

for q_id, data in data_wContents.items():
    data["ideal_answer"] = data_clean[q_id]["ideal_answer"]

with open("training13b_inflated_clean_wContents_IA.jsonl", "w") as f:
    for d in data_wContents.values():
        f.write(f"{json.dumps(d)}\n")
# data_wContents = { x["id"]:x for x in map(json.loads, f)}

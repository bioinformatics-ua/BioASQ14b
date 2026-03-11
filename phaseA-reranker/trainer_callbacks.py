from transformers import TrainerCallback
import torch
from tqdm import tqdm
# from optimum.bettertransformer import BetterTransformer
from collator import RankingCollator
from collections import defaultdict


class ResampleByRerankerCallback(TrainerCallback):
    
    def __init__(self, 
                 train_dataset,
                 tokenizer,
                 start_epoch,
                 interval,
                 num_high_confidence_to_remove,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.start_epoch = start_epoch
        self.interval = interval
        self.num_high_confidence_to_remove = num_high_confidence_to_remove
        self.train_dataset = train_dataset
        self.tokenizer = tokenizer
        self.internal_counter = 0
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        
        if state.epoch > self.start_epoch:
            
            if self.internal_counter > self.interval:
                model = kwargs["model"]
                reranked_negatives = run_inference_on_negatives(model, self.train_dataset.dataset, self.tokenizer)
                reranked_negatives_cut = {qid : {docid: score for docid, score in sorted(docs.items(), key=lambda x: x[1], reverse=True)[self.num_high_confidence_to_remove:] if score > 0.01}
                                        for qid, docs in reranked_negatives.items()}
                
                
                # remove easy training negatives
                for qid, qdata in self.train_dataset.dataset.items():
                    qdata["neg_docs"] = [doc for doc in qdata["neg_docs"] if doc["id"] in reranked_negatives_cut[qdata["id"]]]
                self.internal_counter = -1
                
            self.internal_counter +=1
            
class FlatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = []

        for qid, qdata in tqdm(dataset.items()):
            for doc in qdata["neg_docs"][:100]:
                self.dataset.append({**tokenizer(qdata["question"], doc["text"], truncation=True, max_length=tokenizer.model_max_length),
                        "id": qid,
                        "doc_id": doc["id"]})
                    
    def __getitem__(self, idx): 
        return self.dataset[idx]
    
    def __len__(self):
        return len(self.dataset)
    

def run_inference_on_negatives(model, dataset, tokenizer):
    
    ds_for_inference = FlatDataset(dataset, tokenizer)
    
    # model_inference = BetterTransformer.transform(model, keep_original_model=True)
    model_inference = torch.compile(model) 
    test_dl = torch.utils.data.DataLoader(ds_for_inference, 
                            batch_size=128,
                            collate_fn = RankingCollator(tokenizer=tokenizer))

    run_dict = defaultdict(dict)

    with torch.no_grad():
        for sample in tqdm(test_dl):
            logits = model_inference(**sample["inputs"].to("cuda")).logits
            doc_score = torch.nn.functional.softmax(logits, dim=-1)[:,1].cpu()
            for i in range(doc_score.shape[0]):
                run_dict[sample["id"][i]][sample["doc_id"][i]] = doc_score[i].item()
                
    return run_dict
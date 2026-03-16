"""
Simple smoke test for all models in run_experiments.py.

Verifies that each model:
  1. Loads (tokenizer + AutoModelForSequenceClassification)
  2. Runs a minimal forward pass without error

Run from refactored-trainer directory:
    python test_models.py

No data files required. Uses dummy queries/docs.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS_TO_TEST = [
    "ncbi/MedCPT-Cross-Encoder",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "dmis-lab/biobert-base-cased-v1.2",
    "emilyalsentzer/Bio_ClinicalBERT",
    "michiyasunaga/BioLinkBERT-base",
    "pritamdeka/S-PubMedBert-MS-MARCO",
    "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "bionlp/bluebert_pubmed_uncased_L-12_H-768_A-12",
    "monologg/biobert_v1.1_pubmed",
    "UFNLP/gatortron-base",
    "nboost/pt-biobert-base-msmarco",
    "allenai/specter2_base",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/ms-marco-electra-base",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-v2-m3",
    "jinaai/jina-reranker-v1-base-en",
    "Alibaba-NLP/gte-reranker-base",
    "mixedbread-ai/mxbai-rerank-base-v1",
    "cross-encoder/nli-deberta-v3-base",
]

# Dummy inputs for forward pass (query + document, like in reranking)
SAMPLE_QUERIES = [
    "What is the role of BRCA1 in breast cancer?",
    "How does insulin regulate blood glucose?",
]
SAMPLE_DOCS = [
    "BRCA1 is a tumor suppressor gene associated with hereditary breast and ovarian cancer.",
    "Insulin is a hormone produced by the pancreas that helps cells absorb glucose.",
]


def test_model(model_name: str, device: str = "cpu") -> tuple[bool, str]:
    """Load model and run a minimal forward pass. Returns (success, message)."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        model.to(device)
        model.eval()

        # Minimal forward pass: pair query + doc for each sample
        pairs = [
            f"{q} [SEP] {d}"
            for q, d in zip(SAMPLE_QUERIES, SAMPLE_DOCS)
        ]
        # Some models use different separators; try [SEP] first, tokenizer will adapt
        # If tokenizer has sep_token, it may override. Most handle "[SEP]" or similar.
        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            out = model(**encoded)

        # Expect logits of shape (batch_size, 1) or (batch_size,) for num_labels=1
        logits = out.logits if hasattr(out, "logits") else out[0]
        assert logits.shape[0] == len(pairs), f"Expected batch_size={len(pairs)}, got {logits.shape[0]}"

        return True, "OK"

    except Exception as e:
        return False, str(e)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Smoke test for reranker models")
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Test only the first N models (for quick validation)",
    )
    args = parser.parse_args()

    models = MODELS_TO_TEST[: args.limit] if args.limit else MODELS_TO_TEST
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    print(f"Testing {len(models)} model(s) (load + forward pass)...\n")

    passed = []
    failed = []

    for i, model_name in enumerate(models, 1):
        print(f"[{i}/{len(MODELS_TO_TEST)}] {model_name} ... ", end="", flush=True)
        ok, msg = test_model(model_name, device)
        if ok:
            passed.append(model_name)
            print("PASS")
        else:
            failed.append((model_name, msg))
            print("FAIL")
            print(f"       {msg[:120]}{'...' if len(msg) > 120 else ''}")

    print("\n" + "=" * 60)
    print(f"Passed: {len(passed)}/{len(models)}")
    print(f"Failed: {len(failed)}/{len(models)}")
    if failed:
        print("\nFailed models:")
        for name, err in failed:
            print(f"  - {name}")
            print(f"    {err[:200]}{'...' if len(err) > 200 else ''}")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()

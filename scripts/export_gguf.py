"""Export unsloth/gemma-4-31B + LoRA adapter to GGUF.

Usage:
    uv run python scripts/export_gguf.py \
        --adapter-path /root/lora/snippet_extraction/lora_output/final_adapter \
        --output-dir /root/lora/snippet_extraction/gguf \
        --quant bf16

If --adapter-path is omitted the base model is exported without any adapter.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Gemma-4-31B (+LoRA) to GGUF via Unsloth")
    p.add_argument(
        "--base-model",
        default="unsloth/gemma-4-31B",
        help="HuggingFace model id or local path (default: unsloth/gemma-4-31B)",
    )
    p.add_argument(
        "--adapter-path",
        default=None,
        help="Path to the saved LoRA adapter directory (optional)",
    )
    p.add_argument(
        "--output-dir",
        default="data/training/snippet_extraction/gguf",
        help="Directory where the GGUF file(s) will be written",
    )
    p.add_argument(
        "--quant",
        default="q4_k_m",
        help="Quantization method: q4_k_m, q8_0, f16, … (default: q4_k_m)",
    )
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Max sequence length used when loading the model (default: 2048)",
    )
    p.add_argument(
        "--no-4bit",
        action="store_true",
        help="Load in full precision instead of 4-bit (uses much more VRAM)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    load_in_4bit = not args.no_4bit

    print(f"Loading base model  : {args.base_model}")
    print(f"4-bit quantisation  : {load_in_4bit}")
    print(f"Max seq length      : {args.max_seq_length}")
    print(f"Adapter             : {args.adapter_path or '(none)'}")
    print(f"Output dir          : {output_dir}")
    print(f"GGUF quant method   : {args.quant}")
    print()

    from unsloth import FastModel

    # ------------------------------------------------------------------
    # 1. Load model (+ adapter if provided)
    #
    # Unsloth's save_pretrained_gguf only recognises adapters that were
    # attached through Unsloth itself.  The correct way to load a saved
    # adapter is to pass its directory directly as model_name: Unsloth
    # reads adapter_config.json, loads the base model, and re-attaches
    # the adapter in its own PEFT format — ready for GGUF export.
    # ------------------------------------------------------------------
    if args.adapter_path:
        adapter_path = Path(args.adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter directory not found: {adapter_path}")
        load_target = str(adapter_path)
        print(f"Loading base model + adapter via Unsloth from {adapter_path} …")
    else:
        load_target = args.base_model
        print(f"Loading base model {args.base_model} …")

    model, tokenizer = FastModel.from_pretrained(
        model_name=load_target,
        max_seq_length=args.max_seq_length,
        dtype=None,  # auto-detect bf16 / fp16
        load_in_4bit=load_in_4bit,
        full_finetuning=False,
    )
    print("Model loaded.")

    # ------------------------------------------------------------------
    # 3. Export to GGUF
    # ------------------------------------------------------------------
    print(f"Exporting to GGUF ({args.quant}) → {output_dir} …")
    model.save_pretrained_gguf(str(output_dir), tokenizer, quantization_method=args.quant)
    print("Done. GGUF file(s) written to:", output_dir)


if __name__ == "__main__":
    main()

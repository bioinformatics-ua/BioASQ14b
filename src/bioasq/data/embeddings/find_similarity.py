"""
Multi-GPU similarity computation. Distributes (file0, file1) pairs across
available GPUs via torch.multiprocessing.spawn.
"""

import gc
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer
from tqdm import tqdm

app = typer.Typer()

GC_INTERVAL = 5  # Run gc/empty_cache every N inner iterations to reduce overhead


def load_vectors(path: Path, device: str, pin_memory: bool = False) -> torch.Tensor:
    """Load .npy file to GPU. Uses from_numpy to avoid extra copy."""
    arr = np.load(path, allow_pickle=False)
    t = torch.from_numpy(arr).to(dtype=torch.float16)
    return t.pin_memory().to(device, non_blocking=True) if pin_memory else t.to(device)


def _run_worker(
    rank: int,
    worker_pairs: list[list[tuple[Path, Path]]],
    output_dir: Path,
    T: float,  # noqa: N803
    pin_memory: bool,
) -> None:
    """Entry point for spawn - must be at module level for pickling."""
    worker_process(
        rank=rank,
        pairs=worker_pairs[rank],
        output_dir=output_dir,
        T=T,
        pin_memory=pin_memory,
    )


def worker_process(
    rank: int,
    pairs: list[tuple[Path, Path]],
    output_dir: Path,
    T: float,  # noqa: N803
    pin_memory: bool,
) -> None:
    """Process a partition of (file0, file1) pairs on a single GPU."""
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    inner_count = 0
    for file0_path, file1_path in tqdm(
        pairs,
        desc=f"GPU {rank}",
        position=rank,
        leave=False,
    ):
        file0_range: tuple[str, str] = tuple(file0_path.stem.split("_")[1:3])
        file1_range: tuple[str, str] = tuple(file1_path.stem.split("_")[1:3])

        output_file = output_dir / (
            f"shard_T{T}_{file0_range[0]}-{file0_range[1]}_{file1_range[0]}-{file1_range[1]}.npy"
        )

        if output_file.exists():
            continue

        file0 = load_vectors(file0_path, device, pin_memory=pin_memory)
        file1 = load_vectors(file1_path, device, pin_memory=pin_memory)

        with torch.inference_mode():
            sim_docs = file0 @ file1.T
            if file0_range == file1_range:
                sim_docs = torch.triu(sim_docs, diagonal=1)

            mask = sim_docs > T
            values = sim_docs[mask]
            sim_docs_indices = torch.argwhere(mask)
            combined = np.hstack(
                [
                    sim_docs_indices.cpu().numpy(),
                    values.cpu().numpy().reshape(-1, 1),
                ]
            )

        np.save(output_file, combined, allow_pickle=False)

        del file0, file1, sim_docs, values, sim_docs_indices, mask

        inner_count += 1
        if inner_count % GC_INTERVAL == 0:
            torch.cuda.empty_cache()
            gc.collect()


@app.command()
def main(
    dense_vectors_dir: Annotated[Path, typer.Argument(help="Dense vectors directory.")] = Path(
        "../dense_vectors_numpy"
    ),
    T: Annotated[float, typer.Option("-t", "--threshold", help="Threshold.")] = 0.80,  # noqa: N803
    output_dir: Annotated[
        Path, typer.Option("-o", "--output-dir", help="Results directory.")
    ] = Path("../similarity_results"),
    pin_memory: Annotated[
        bool, typer.Option("--pin-memory", help="Use pinned CPU memory for faster H2D transfer.")
    ] = True,
    n_gpus: Annotated[
        int | None,
        typer.Option("--n-gpus", "-n", help="Number of GPUs to use. Default: all available."),
    ] = None,
) -> None:
    vector_files = sorted(dense_vectors_dir.glob("*.npy"), key=lambda x: int(x.stem.split("_")[1]))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build all (file0, file1) pairs (upper triangle incl. diagonal blocks)
    all_pairs: list[tuple[Path, Path]] = []
    for file_i, file0_path in enumerate(vector_files):
        for file1_path in vector_files[file_i:]:
            f0_start, f0_end = file0_path.stem.split("_")[1:3]
            file0_range = (f0_start, f0_end)
            f1_start, f1_end = file1_path.stem.split("_")[1:3]
            file1_range = (f1_start, f1_end)
            output_file = output_dir / (
                f"shard_T{T}_{file0_range[0]}-{file0_range[1]}"
                f"_{file1_range[0]}-{file1_range[1]}.npy"
            )
            if output_file.exists():
                continue
            all_pairs.append((file0_path, file1_path))

    if not all_pairs:
        typer.echo("All pairs already processed. Nothing to do.")
        return

    n_available = torch.cuda.device_count()
    if n_available == 0:
        typer.echo("No CUDA GPUs available. Exiting.")
        raise typer.Exit(1)

    n_workers = n_gpus if n_gpus is not None else n_available
    n_workers = min(n_workers, n_available, len(all_pairs))

    typer.echo(
        f"Processing {len(all_pairs)} pairs on {n_workers} GPU(s) "
        f"(skipping {len(vector_files) * (len(vector_files) + 1) // 2 - len(all_pairs)} existing)."
    )

    # Partition pairs across workers: worker r gets pairs where i % n_workers == r
    worker_pairs: list[list[tuple[Path, Path]]] = [[] for _ in range(n_workers)]
    for i, pair in enumerate(all_pairs):
        worker_pairs[i % n_workers].append(pair)

    # Ensure torch multiprocessing uses spawn (required for CUDA)
    torch.multiprocessing.set_start_method("spawn", force=True)

    torch.multiprocessing.spawn(
        _run_worker,
        args=(worker_pairs, output_dir, T, pin_memory),
        nprocs=n_workers,
        join=True,
    )


if __name__ == "__main__":
    app()

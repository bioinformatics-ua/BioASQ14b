import gc
from pathlib import Path
from tqdm import tqdm
import typer
import numpy as np
import torch

app = typer.Typer()

DEVICE = "cuda:1"
GC_INTERVAL = 5  # Run gc/empty_cache every N inner iterations to reduce overhead


def load_vectors(path: Path, pin_memory: bool = False) -> torch.Tensor:
    """Load .npy file to GPU. Uses from_numpy to avoid extra copy. pin_memory enables faster H2D transfer."""
    arr = np.load(path, allow_pickle=False)
    t = torch.from_numpy(arr).to(dtype=torch.float16)
    if pin_memory:
        t = t.pin_memory().to(DEVICE, non_blocking=True)
    else:
        t = t.to(DEVICE)
    return t


@app.command()
def main(
    dense_vectors_dir: Path = typer.Argument(
        Path("../dense_vectors"), help="Dense vectors directory."
    ),
    T: float = typer.Option(0.80, "-t", "--threshold", help="Threshold."),
    output_dir: Path = typer.Option(
        Path("../similarity_results"), "-o", "--output-dir", help="Results directory."
    ),
    pin_memory: bool = typer.Option(
        False, "--pin-memory", help="Use pinned CPU memory for faster H2D transfer."
    ),
):
    vector_files = sorted(
        dense_vectors_dir.glob("*.npy"), key=lambda x: int(x.stem.split("_")[1])
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_i, file0_path in tqdm(
        enumerate(vector_files),
        desc="Processing row files",
        unit="file",
        position=0,
        total=len(vector_files),
    ):
        file0_range: tuple[str, str] = tuple(file0_path.stem.split("_")[1:3])
        file0 = load_vectors(file0_path, pin_memory=pin_memory)

        inner_count = 0
        for file1_path in tqdm(
            vector_files[file_i:],
            desc="Processing col files",
            unit="file",
            position=1,
        ):
            file1_range: tuple[str, str] = tuple(file1_path.stem.split("_")[1:3])
            file1 = load_vectors(file1_path, pin_memory=pin_memory)

            output_file = (
                output_dir
                / f"shard_T{T}_{file0_range[0]}-{file0_range[1]}_{file1_range[0]}-{file1_range[1]}.npy"
            )

            # Skip processing if output already exists
            if output_file.exists():
                continue

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

            # np.save is faster than torch.save for raw arrays and produces smaller files
            np.save(
                output_file,
                combined,
                allow_pickle=False,
            )

            del file1, sim_docs, values, sim_docs_indices, mask

            inner_count += 1
            if inner_count % GC_INTERVAL == 0:
                torch.cuda.empty_cache()
                gc.collect()


if __name__ == "__main__":
    app()

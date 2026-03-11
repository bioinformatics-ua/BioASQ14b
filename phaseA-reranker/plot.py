import orjson
from pathlib import Path

import matplotlib.pyplot as plt

LOG_PATH = Path(
    "trained_models_tmp/nvidia-llama-nemotron-rerank-1b-v2-42-E10-SBasicSampler-SPCausalLMSamplePreprocessing-val-quality_data-CBFalse-KN1-GA1-ExPOSFalse-warmupFalse/checkpoint-25490/trainer_state.json"
)


def parse_log_history(path: Path) -> list[dict[str, int | float]]:
    with open(path, "rb") as f:
        log_history: list[dict[str, int | float]] | dict[str, int | float] = (
            orjson.loads(f.read())
        )

    if isinstance(log_history, dict):
        log_history: list[dict[str, int | float]] = log_history.get("log_history", [])

    return log_history


def load_metrics(log_history: list[dict[str, int | float]]) -> tuple[dict, dict]:
    train: dict[str, list[float]] = {"epoch": [], "loss": []}
    val: dict[str, list[float]] = {"epoch": [], "loss": [], "dice": [], "iou": []}

    for entry in log_history:
        epoch = entry.get("epoch")
        if epoch is None:
            continue
        if "eval_loss" in entry:
            val["epoch"].append(epoch)
            val["loss"].append(entry["eval_loss"])
            val["dice"].append(entry.get("eval_dice", float("nan")))
            val["iou"].append(entry.get("eval_iou", float("nan")))
        elif "loss" in entry and "eval_loss" not in entry:
            train["epoch"].append(epoch)
            train["loss"].append(entry["loss"])

    return train, val


def plot(train: dict, val: dict, save_dir: Path = Path("outputs/plots")) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Loss ---
    ax = axes[0]
    ax.plot(train["epoch"], train["loss"], label="Train")
    ax.plot(val["epoch"], val["loss"], label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Dice ---
    ax = axes[1]
    ax.plot(val["epoch"], val["dice"], color="tab:green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_title("Dice Score")
    ax.grid(True, alpha=0.3)

    # --- IoU ---
    ax = axes[2]
    ax.plot(val["epoch"], val["iou"], color="tab:orange")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("IoU Score")
    ax.grid(True, alpha=0.3)

    fig.suptitle("UNet Training — Kvasir-SEG", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "training_plots.png", dpi=150)
    plt.show()
    print(f"Saved to {save_dir / 'training_plots.png'}")


if __name__ == "__main__":
    log_history = parse_log_history(LOG_PATH)
    train, val = load_metrics(log_history)
    plot(train, val)

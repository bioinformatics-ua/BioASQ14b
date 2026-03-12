from huggingface_hub import HfApi
import os

# Define Hugging Face username and repository
HF_USERNAME = "IEETA"
MODEL_REPO = "BioASQ-13B"  # Change this to your actual repository name

# Path to the trained models
MODELS_DIR = "trained_models_b02"

# Initialize Hugging Face API
api = HfApi()

# Ensure the repository exists
repo_id = f"{HF_USERNAME}/{MODEL_REPO}"
api.create_repo(repo_id=repo_id, exist_ok=True)


# Helper function to extract checkpoint number
def extract_checkpoint_number(checkpoint_name):
    """Extracts checkpoint number from name (assumes format 'checkpoint-XXXXX')."""
    parts = checkpoint_name.split("-")
    for part in parts:
        if part.isdigit():
            return int(part)
    return -1  # Default if no number is found


# Dictionary to store latest checkpoint per model
latest_checkpoints = {}

# Iterate through model folders
for model_name in os.listdir(MODELS_DIR):
    model_path = os.path.join(MODELS_DIR, model_name)

    if os.path.isdir(model_path):  # Ensure it's a directory and allowed model
        # Find all checkpoint subdirectories
        checkpoints = [d for d in os.listdir(model_path) if d.startswith("checkpoint-")]
        checkpoints = sorted(
            checkpoints, key=extract_checkpoint_number, reverse=True
        )  # Sort by latest

        if checkpoints:  # If there are checkpoints, take the latest allowed one
            latest_checkpoint = checkpoints[0]
            latest_checkpoint_number = extract_checkpoint_number(latest_checkpoint)

            # Ensure the checkpoint is the one in the allowed list
            # if latest_checkpoint_number == ALLOWED_MODELS[model_name]:
            latest_checkpoints[model_name] = os.path.join(model_path, latest_checkpoint)

# Upload the latest checkpoint for each allowed model
for model_name, checkpoint_path in latest_checkpoints.items():
    checkpoint_name = os.path.basename(checkpoint_path)
    revision_name = (
        f"{model_name}-{checkpoint_name}"  # Format: modelname_checkpoint-XXXXX
    )

    print(
        f"Uploading latest checkpoint {checkpoint_name} from {model_name} to {repo_id} as revision {revision_name}..."
    )

    try:
        api.create_branch(repo_id=repo_id, branch=revision_name)

        api.upload_folder(
            folder_path=checkpoint_path,
            repo_id=repo_id,
            revision=revision_name,
            ignore_patterns="optimizer.pt",
        )
    except:
        print("erro")
    print(f"Uploaded {checkpoint_name} successfully as {revision_name}!\n")

from huggingface_hub import HfApi

api = HfApi()

print("Uploading heavy models to the unlimited Model Hub...")

api.upload_folder(
    folder_path="checkpoints",         # Grab ONLY the checkpoints folder
    path_in_repo="checkpoints",        # Keep the same folder structure
    repo_id="faiz06/hybrid-emotion-weights", # Your new Model repo
    repo_type="model",                 # Specify it's a model, not a space!
)

print("Upload complete! Weights are safely stored.")
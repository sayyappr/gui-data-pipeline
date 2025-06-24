import os
import zipfile
from huggingface_hub import snapshot_download

DATASETS = [
    {
        "repo_id": "OS-Copilot/OS-Atlas-data",
        "local_dir": "data/osatlas",
    },
    {
        "repo_id": "xlangai/aguvis-stage1",
        "local_dir": "data/aguvis_stage1",
    },
    {
        "repo_id": "xlangai/aguvis-stage2",
        "local_dir": "data/aguvis_stage2",
    }
]

def download_and_unzip(dataset):
    repo_id = dataset["repo_id"]
    target_dir = dataset["local_dir"]

    print(f"\nDownloading {repo_id} into {target_dir}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        repo_type="dataset",
        ignore_patterns=["*.md", "*.txt"]
    )
    print("✅ Downloaded.")

    print(f"Unzipping .zip files in {target_dir}...")
    for root, _, files in os.walk(target_dir):
        for fname in files:
            if fname.endswith(".zip"):
                zip_path = os.path.join(root, fname)
                try:
                    print(f" Unzipping {zip_path}")
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(root)
                    os.remove(zip_path)
                    print(f"Unzipped and deleted {zip_path}")
                except Exception as e:
                    print(f"Failed to unzip {zip_path}: {e}")

# Run for all datasets
for ds in DATASETS:
    download_and_unzip(ds)

print("\n All datasets downloaded, unzipped, and cleaned up.")

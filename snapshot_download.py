import os
import time
from huggingface_hub import snapshot_download, hf_hub_url, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError


def download_with_retries(repo_id, local_dir, token=None, max_retries=3):
    os.makedirs(local_dir, exist_ok=True)

    print(f"Starting download for repo '{repo_id}' into '{local_dir}'")

    for attempt in range(1, max_retries + 1):
        try:
            # Download snapshot (all files)
            snapshot_download(repo_id=repo_id, local_dir=local_dir, token=token)
            print("Snapshot download complete.")
        except Exception as e:
            print(f"Download attempt {attempt} failed with error: {e}")
            if attempt == max_retries:
                raise
            print(f"Retrying in 5 seconds...")
            time.sleep(5)
            continue

        # Check if all files exist by comparing to repo file list
        from huggingface_hub import HfApi
        api = HfApi()

        try:
            repo_files = [f.rfilename for f in api.list_repo_files(repo_id, token=token)]
        except RepositoryNotFoundError:
            print("Repo not found or token missing permission.")
            return

        missing_files = [f for f in repo_files if not os.path.isfile(os.path.join(local_dir, f))]
        if not missing_files:
            print("All files downloaded successfully!")
            break
        else:
            print(f"Missing files after attempt {attempt}: {missing_files}")
            if attempt == max_retries:
                raise FileNotFoundError(f"Missing files after {max_retries} attempts: {missing_files}")
            print(f"Retrying missing files download in 5 seconds...")
            time.sleep(5)
            # Optional: Download missing files one by one
            for mf in missing_files:
                file_path = os.path.join(local_dir, mf)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                try:
                    hf_hub_download(repo_id, mf, token=token, cache_dir=local_dir)
                    print(f"Downloaded missing file: {mf}")
                except Exception as e:
                    print(f"Failed to download {mf}: {e}")

if __name__ == "__main__":
    import os

    HF_TOKEN = ""
    print("JAVA_HOME:", os.environ)

    REPO_ID = "mistralai/Mistral-7B-Instruct-v0.2"
    LOCAL_MODEL_DIR = "models/llm/mistral-7b"

    download_with_retries(REPO_ID, LOCAL_MODEL_DIR, token=HF_TOKEN)

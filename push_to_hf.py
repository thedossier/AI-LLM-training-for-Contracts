from huggingface_hub import HfApi, login
import os

def upload_to_hf_hub(repo_id):
    # Initialize HfApi
    api = HfApi()

    # Define the files to upload
    files_to_upload = ["data/train.csv", "data/test.csv", "data/README.md"]
    uploaded_files = []

    # Upload each file if it exists
    for file_path in files_to_upload:
        if os.path.exists(file_path):
            print(f"Uploading {file_path}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=os.path.basename(file_path),  # Only the filename, not the path
                repo_id=repo_id,
                repo_type="dataset",
            )
            print(f"Uploaded {file_path}.")
            uploaded_files.append(file_path)
        else:
            print(f"{file_path} does not exist, skipping.")

    # Summary
    print("\nSummary:")
    if uploaded_files:
        print("Uploaded files:")
        for file in uploaded_files:
            print(f"- {file}")
    else:
        print("No files were uploaded.")

def main():
    # Login
    print("Logging in to Hugging Face account...")
    login()

    # Get repo path from the user
    repo_id = input("Enter the path to the Hugging Face dataset repo (e.g. username/repo_name): ")

    # Upload files to the repo
    upload_to_hf_hub(repo_id)

if __name__ == "__main__":
    main()

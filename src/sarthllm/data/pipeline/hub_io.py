# Pushes or pulls the built token shards in artifacts/ to/from a Hugging Face dataset repo.
# Shards are .bin + meta.json, so the repo is only read back by TokenShards, never trained from directly.
import argparse

from huggingface_hub import HfApi, snapshot_download


def push(folder, repo_id, private=True):
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=repo_id, repo_type="dataset")


def pull(repo_id, local_dir):
    return snapshot_download(repo_id, repo_type="dataset", local_dir=local_dir)


def main(args):
    if args.pull:
        print(pull(args.repo_id, args.local_dir or args.folder))
    else:
        push(args.folder, args.repo_id, private=not args.public)
        print(f"pushed {args.folder} -> {args.repo_id}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("repo_id")
    p.add_argument("--folder", default="artifacts/data/fineweb_edu")
    p.add_argument("--local-dir", dest="local_dir")
    p.add_argument("--pull", action="store_true", help="download instead of upload")
    p.add_argument("--public", action="store_true", help="make the repo public on first push")
    main(p.parse_args())
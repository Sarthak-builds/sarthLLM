# Syncs checkpoints with a private Hugging Face model repo so a new Kaggle session can resume training.
import time
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

from sarthllm.train.checkpoint import list_checkpoints


def with_retries(action, attempts=4, first_delay=5):
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as error:
            if attempt == attempts:
                raise
            delay = first_delay * 2 ** (attempt - 1)
            print(f"hub error: {error!r}; retry {attempt}/{attempts - 1} in {delay}s")
            time.sleep(delay)


def list_remote_files(api, repo_id):
    return with_retries(lambda: api.list_repo_files(repo_id, repo_type="model"))


def newest_checkpoint_name(files):
    names = sorted(f for f in files if f.startswith("ckpt_") and f.endswith(".pt"))
    return names[-1] if names else None


def pull_latest(out_dir, repo_id):
    # fails loudly on a bad or read-only token: better to stop now than after hours of training
    api = HfApi()
    with_retries(api.whoami)
    with_retries(lambda: api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True))

    out_dir = Path(out_dir)
    files = list_remote_files(api, repo_id)
    remote_newest = newest_checkpoint_name(files)
    if remote_newest is None:
        print(f"no checkpoint in {repo_id}: starting a fresh run")
        return
    local = list_checkpoints(out_dir)
    if local and local[-1].name >= remote_newest:
        print("local checkpoint is already up to date")
        return

    print(f"downloading {remote_newest} from {repo_id}")
    with_retries(lambda: hf_hub_download(repo_id, remote_newest, repo_type="model", local_dir=out_dir))
    if "metrics.jsonl" in files and not (out_dir / "metrics.jsonl").exists():
        with_retries(lambda: hf_hub_download(repo_id, "metrics.jsonl", repo_type="model", local_dir=out_dir))


def push_run(out_dir, repo_id, include_best=False):
    # never raises: a failed upload must not stop training, the next upload tries again
    try:
        _push_run(out_dir, repo_id, include_best)
        return True
    except Exception as error:
        print(f"WARNING: upload to {repo_id} failed ({error!r}); training continues")
        return False


def _push_run(out_dir, repo_id, include_best):
    api = HfApi()
    out_dir = Path(out_dir)
    checkpoints = list_checkpoints(out_dir)
    if not checkpoints:
        return
    newest = checkpoints[-1]

    files = list_remote_files(api, repo_id)
    remote_newest = newest_checkpoint_name(files)
    if remote_newest and remote_newest > newest.name:
        print(f"{repo_id} has {remote_newest}, newer than local {newest.name}: not overwriting")
        return

    def upload(path):
        with_retries(lambda: api.upload_file(
            path_or_fileobj=str(path), path_in_repo=path.name, repo_id=repo_id, repo_type="model"
        ))

    if remote_newest != newest.name:  # upload the new checkpoint first, delete old ones afterwards
        print(f"uploading {newest.name} to {repo_id}")
        upload(newest)
    metrics = out_dir / "metrics.jsonl"
    if metrics.exists():
        upload(metrics)
    best = out_dir / "best.pt"
    if include_best and best.exists():
        upload(best)

    old_checkpoints = [f for f in files if f.startswith("ckpt_") and f.endswith(".pt") and f < newest.name]
    for name in old_checkpoints:
        with_retries(lambda: api.delete_file(name, repo_id=repo_id, repo_type="model"))
    if old_checkpoints or include_best:  # drop old file versions so storage does not pile up
        try:
            api.super_squash_history(repo_id, repo_type="model")
        except Exception as error:
            print(f"could not squash history ({error!r}); harmless")
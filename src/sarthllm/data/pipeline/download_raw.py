# Downloads raw parquet files from Hugging Face dataset based on configs/data.yaml
import argparse

import yaml
from huggingface_hub import HfApi, hf_hub_download


def main(cfg_path: str) -> None:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    tree = HfApi().list_repo_tree(
        cfg["dataset"], path_in_repo=cfg["raw_prefix"].rstrip("/"), repo_type="dataset"
    )
    files = sorted((f for f in tree if f.path.endswith(".parquet")), key=lambda f: f.path)
    if not files:
        raise SystemExit(f"No parquet files under {cfg['raw_prefix']}. Check the repo's Files tab.")

    n = cfg["raw_files"]
    print(f"{len(files)} files available; downloading first {n}")
    for f in files[:n]:
        print(f"-> {f.path} ({f.size / 1e9:.2f} GB)")
        hf_hub_download(cfg["dataset"], f.path, repo_type="dataset", local_dir=cfg["raw_dir"])
    print("done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/data.yaml")
    main(p.parse_args().config)

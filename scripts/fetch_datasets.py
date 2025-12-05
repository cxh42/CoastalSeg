import os
import sys
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

# Datasets used for training
DATASETS = {
    "AveMujica/CoastalSeg-MM": "SegmentModelTraining/MetalMarcy/dataset",
    "AveMujica/CoastalSeg-SJ": "SegmentModelTraining/SilhouetteJaenette/dataset",
}

HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def prune_unpaired_images(local_dir: str | Path) -> None:
    local_dir = Path(local_dir)
    removed = 0
    for split in ("train", "valid", "test"):
        sdir = local_dir / split
        if not sdir.exists():
            continue
        img_paths = list(sdir.glob("*.jpg")) + list(sdir.glob("*.jpeg")) + list(sdir.glob("*.png"))
        img_paths = [p for p in img_paths if not p.name.endswith("_mask.png")]
        mask_bases = {p.stem[:-5] if p.stem.endswith("_mask") else p.stem for p in sdir.glob("*_mask.png")}
        for ip in img_paths:
            base = ip.stem
            if base not in mask_bases:
                try:
                    ip.unlink()
                    removed += 1
                    print(f"[dataset] Removed unpaired image: {ip}")
                except Exception as e:
                    print(f"[dataset] Warning: failed to remove {ip}: {e}")
    if removed:
        print(f"[dataset] Pruned {removed} unpaired images without masks")


def fetch_datasets(force: bool = False):
    for repo_id, local_dir in DATASETS.items():
        print(f"[dataset] Sync {repo_id} -> {local_dir}")
        ld = Path(local_dir)
        if force and ld.exists():
            print(f"[dataset] Removing existing directory: {ld}")
            shutil.rmtree(ld)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=HF_TOKEN,
        )
        for split in ("train", "valid", "test"):
            p = Path(local_dir) / split
            print(f"[dataset] {repo_id}:{split} -> {p} {'OK' if p.exists() else 'MISSING'}")
        prune_unpaired_images(local_dir)


def main(argv: list[str]) -> int:
    force = "--force" in argv
    fetch_datasets(force=force)
    print("Datasets downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

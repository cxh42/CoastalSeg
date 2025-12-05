import os
import sys
from pathlib import Path
from typing import Optional

import requests

# Model weights and reference vectors (for outlier detection)
MODELS = {
    "MM_best_model.pth": "https://huggingface.co/AveMujica/CoastalSeg-MM/resolve/main/MM_best_model.pth",
    "SJ_best_model.pth": "https://huggingface.co/AveMujica/CoastalSeg-SJ/resolve/main/SJ_best_model.pth",
}

REF_VECTORS = {
    "MM_mean.npy": "https://huggingface.co/spaces/AveMujica/CoastalSegment/resolve/main/models/MM_mean.npy",
    "SJ_mean.npy": "https://huggingface.co/spaces/AveMujica/CoastalSegment/resolve/main/models/SJ_mean.npy",
}

HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


def http_download(url: str, dest: Path, chunk: int = 1 << 20, token: Optional[str] = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else None
    with requests.get(url, stream=True, timeout=300, headers=headers) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for b in r.iter_content(chunk_size=chunk):
                if b:
                    f.write(b)


def fetch_models(force: bool = False):
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    for fname, url in MODELS.items():
        out = models_dir / fname
        if out.exists() and not force:
            print(f"[models] Exists: {out}")
            continue
        print(f"[models] Downloading {fname} ...")
        http_download(url, out, token=HF_TOKEN)
        print(f"[models] Saved: {out} ({out.stat().st_size} bytes)")

    # Optional reference vectors
    for fname, url in REF_VECTORS.items():
        out = models_dir / fname
        if out.exists() and not force:
            print(f"[models] Exists: {out}")
            continue
        try:
            print(f"[models] Downloading {fname} ...")
            http_download(url, out, token=HF_TOKEN)
            print(f"[models] Saved: {out} ({out.stat().st_size} bytes)")
        except Exception as e:
            # Non-fatal if vectors are unavailable
            print(f"[models] Warning: failed to download {fname}: {e}")


def main(argv: list[str]) -> int:
    force = "--force" in argv
    fetch_models(force=force)
    print("Models downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

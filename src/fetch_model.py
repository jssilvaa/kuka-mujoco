from __future__ import annotations

from pathlib import Path
import json
import urllib.request


GITHUB_API = "https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents"
RAW_BASE = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main"


def _list_dir(path: str):
    url = f"{GITHUB_API}/{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def _download_file(src_rel: str, dst: Path) -> None:
    url = f"{RAW_BASE}/{src_rel}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dst)


def fetch_model(model_dir: str, out_root: Path) -> None:
    entries = _list_dir(model_dir)
    for item in entries:
        name = item["name"]
        rel = f"{model_dir}/{name}"
        if item["type"] == "file":
            if name.endswith((".xml", ".png", ".obj")):
                _download_file(rel, out_root / model_dir / name)
        elif item["type"] == "dir" and name in ("assets",):
            subentries = _list_dir(rel)
            for sub in subentries:
                if sub["type"] == "file":
                    _download_file(f"{rel}/{sub['name']}", out_root / rel / sub["name"])


if __name__ == "__main__":
    root = Path("data/").resolve()
    model = "kuka_iiwa_14"
    fetch_model(model, root)
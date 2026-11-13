"""Downloads Kronos at pinned versions and checks every file against its SHA-256.

The code (MIT) is one commit of github.com/shiyu-coder/Kronos, which is not a package; the weights
(MIT) are two Hugging Face repositories at pinned revisions. Run it with the sidecar's Python:
`python fetch_kronos.py <folder>`.
"""

import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"
ARCHIVE = f"https://github.com/shiyu-coder/Kronos/archive/{COMMIT}.tar.gz"
ARCHIVE_SHA256 = "969719e47b2134d8a56533784508b6d859bd1c9aacb1b62e4a504cb4fc096021"
CODE = ("model/__init__.py", "model/kronos.py", "model/module.py", "LICENSE")
WEIGHTS = {
    "small": (
        "NeoQuasar/Kronos-small",
        "901c26c1332695a2a8f243eb2f37243a37bea320",
        {
            "config.json": "5e0f6a605d5f81b5c9b559fe5cf716a1acb041c744e6f41bd05b097b7a685396",
            "model.safetensors": "b082dfcbd8e8c142a725c8bbb99781802f38fec81210e13479effb32b3c3e020",
        },
    ),
    "tokenizer": (
        "NeoQuasar/Kronos-Tokenizer-base",
        "0e0117387f39004a9016484a186a908917e22426",
        {
            "config.json": "2366e7ccfec76cbc19cf3c4c1b9c5d901be336ca1e83f2d2292c9bff381b77a2",
            "model.safetensors": "59d85f6af76a2c3b8240ea06cb21db4213b4eeca053f246b23e29cf832fc6bee",
        },
    ),
}


def digest_of(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch_code(folder: Path) -> None:
    with urllib.request.urlopen(ARCHIVE, timeout=60) as response:  # noqa: S310
        blob = response.read()
    if hashlib.sha256(blob).hexdigest() != ARCHIVE_SHA256:
        raise SystemExit("The Kronos archive does not match its pinned SHA-256")
    with tarfile.open(fileobj=io.BytesIO(blob)) as archive:
        for name in CODE:
            member = archive.extractfile(f"Kronos-{COMMIT}/{name}")
            if member is None:
                raise SystemExit(f"{name} is not in the Kronos archive")
            target = folder / "code" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(member.read())


def fetch_weights(folder: Path) -> None:
    for name, (repo, revision, files) in WEIGHTS.items():
        for file, expected in files.items():
            path = Path(hf_hub_download(repo, file, revision=revision, local_dir=folder / name))
            if digest_of(path) != expected:
                path.unlink()
                raise SystemExit(f"{repo}/{file} does not match its pinned SHA-256 and was removed")


def main(target: str) -> None:
    folder = Path(target)
    folder.mkdir(parents=True, exist_ok=True)
    fetch_code(folder)
    fetch_weights(folder)
    record = {"commit": COMMIT, "weights": {k: [v[0], v[1]] for k, v in WEIGHTS.items()}}
    (folder / "revision.json").write_text(json.dumps(record, indent=2))
    sys.stdout.write(f"Kronos {COMMIT[:7]} and its weights are in {folder}, every file checked\n")


if __name__ == "__main__":
    main(sys.argv[1])

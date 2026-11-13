"""Downloads FinBERT at one pinned commit and checks every file against its published SHA-256.

The weights are licensed CC BY-NC-SA 4.0: non-commercial use, with attribution and share-alike.
Run it with the sidecar's Python: `python fetch_finbert.py <folder>`.
"""

import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "ProsusAI/finbert"
REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
FILES = {
    "model.safetensors": "e5897858ff819aad7629b96ce521ae5477952d03634ef5dd30ed2d76357a9f00",
    "config.json": "f6449ddda85eb726207a40be59c0cd3bd4b142ccb27298d5e45f9ae3396b1abe",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    "tokenizer_config.json": "ffc7913f6084b138dad68de5a0c2f0ee25ac983709260901e01f53be5796b9d9",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
}


def digest_of(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main(target: str) -> None:
    folder = Path(target)
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = Path(hf_hub_download(REPO, name, revision=REVISION, local_dir=folder))
        if digest_of(path) != expected:
            path.unlink()
            raise SystemExit(f"{name} does not match its published SHA-256 and was removed")
    record = {"repo": REPO, "revision": REVISION, "sha256": FILES}
    (folder / "revision.json").write_text(json.dumps(record, indent=2))
    sys.stdout.write(f"FinBERT {REVISION[:7]} is in {folder}, every file checked\n")


if __name__ == "__main__":
    main(sys.argv[1])

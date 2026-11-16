"""Download deepset/prompt-injections into data/ as JSONL and record dataset metadata."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from injection_guard.data import DATA_DIR, DATASET_NAME, download  # noqa: E402


def main() -> None:
    download()
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(DATASET_NAME)
    card = info.card_data.to_dict() if info.card_data else {}
    meta = {"dataset": DATASET_NAME, "revision": info.sha, "license": card.get("license")}
    (DATA_DIR / "dataset_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

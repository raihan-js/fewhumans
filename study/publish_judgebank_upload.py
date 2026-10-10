"""Upload the JudgeBank to Hugging Face (needs HF_TOKEN env, owner-provided).

Usage: HF_TOKEN=... .venv/bin/python study/publish_judgebank_upload.py [--dry-run]
Creates/updates raihan-js/fewhumans-judgebank with the 8 parquets + datasheet.
"""

from __future__ import annotations

import os
import sys

DRY = "--dry-run" in sys.argv
REPO = "raihan-js/fewhumans-judgebank"


def main() -> None:
    token = os.environ.get("HF_TOKEN", "")
    if not token and not DRY:
        raise SystemExit("HF_TOKEN env missing (owner-provided, never committed)")
    from huggingface_hub import HfApi

    api = HfApi(token=token or "dry")
    if DRY:
        import glob

        print("dry-run: would upload to", REPO)
        print("\n".join(sorted(glob.glob("data/release/judgebank/*.parquet"))))
        print("data/release/DATASHEET.md -> README.md")
        return
    api.create_repo(REPO, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path="data/release/judgebank",
                      path_in_repo=".", allow_patterns="*.parquet")
    with open("data/release/DATASHEET.md") as f:
        card = f.read()
    header = "---\nlicense: other\ntask_categories:\n- text-classification\nlanguage:\n- en\n---\n"
    api.upload_file(path_or_fileobj=(header + "# FewHumans JudgeBank\n\n" + card).encode(),
                    path_in_repo="README.md", repo_id=REPO, repo_type="dataset")
    print("uploaded", REPO)


if __name__ == "__main__":
    main()

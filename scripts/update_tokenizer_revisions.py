"""Resolve moving Hub refs to pinned commits without downloading model files."""

import argparse
import json
import re
from pathlib import Path

from huggingface_hub import HfApi, get_token
from prepare_tokenizer_assets import retry_hub_request


def update_manifest(path, api, include_token_required=False):
    path = Path(path)
    manifest = json.loads(path.read_text())
    changes = []
    for entry in manifest["models"]:
        # null means the test intentionally targets a particular release.
        ref = entry.get("update_ref", "main")
        if ref is None or (entry["requires_token"] and not include_token_required):
            continue
        model_id = entry["repo_id"]
        info = retry_hub_request(
            lambda: api.model_info(model_id, revision=ref, expand=["sha"]), model_id
        )
        if not isinstance(info.sha, str) or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
            raise ValueError("Invalid Hub commit SHA for " + model_id)
        if info.sha != entry["revision"]:
            changes.append((model_id, entry["revision"], info.sha))
            entry["revision"] = info.sha
    # Write only after every eligible model resolves successfully.
    if changes:
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="tests/tokenizer_assets.json")
    parser.add_argument("--summary", default=".ci/tokenizer-update.md")
    parser.add_argument("--include-token-required", action="store_true")
    args = parser.parse_args()
    changes = update_manifest(
        args.manifest, HfApi(), args.include_token_required or bool(get_token())
    )
    lines = ["Update pinned tokenizer/config revisions to their tracked Hub refs.", ""]
    for model_id, old, new in changes:
        lines.append(f"- `{model_id}`: `{old}` → `{new}`")
    if not changes:
        lines.append("No revision changes detected.")
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines) + "\n")
    print(f"Updated {len(changes)} tokenizer revisions", flush=True)


if __name__ == "__main__":
    main()

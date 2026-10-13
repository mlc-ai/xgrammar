"""Build a portable, credential-free tokenizer bundle for offline CI tests."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import get_token, snapshot_download

# Include custom tokenizer/config implementations and chat templates, but no weights.
ASSET_PATTERNS = ["*.json", "*.model", "*.tiktoken", "*.txt", "*.py", "*.jinja", "*.bpe"]


def prepare_tiktoken_cache(output):
    # Phi-3-small calls get_encoding("cl100k_base") even when loaded from local files.
    # Seed tiktoken's URL-hash cache from the vocabulary in its pinned Hub snapshot.
    output = Path(output)
    phi_vocab = output / "microsoft--Phi-3-small-8k-instruct" / "cl100k_base.tiktoken"
    if phi_vocab.is_file():
        cache = output / "tiktoken"
        cache.mkdir(exist_ok=True)
        url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        shutil.copyfile(phi_vocab, cache / hashlib.sha1(url.encode()).hexdigest())


def prepare(manifest_path, output, include_token_required=False):
    manifest = json.loads(Path(manifest_path).read_text())
    output = Path(output)
    # A public-only rebuild must never retain files from a previous full bundle.
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    index = {}
    for entry in manifest["models"]:
        if entry["requires_token"] and not include_token_required:
            continue
        model_id = entry["repo_id"]
        print("Preparing " + model_id, flush=True)
        # Serial repositories and serial file downloads avoid a download stampede.
        snapshot = Path(
            snapshot_download(
                model_id, revision=entry["revision"], allow_patterns=ASSET_PATTERNS, max_workers=1
            )
        )
        relative = model_id.replace("/", "--")
        destination = output / relative
        if destination.exists():
            shutil.rmtree(destination)
        # Dereference Hub cache symlinks for portable Windows/Linux/macOS artifacts.
        shutil.copytree(snapshot, destination)
        index[model_id] = {"path": relative, "revision": entry["revision"], "commit": snapshot.name}
    prepare_tiktoken_cache(output)
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="tests/tokenizer_assets.json")
    parser.add_argument("--output", default=".ci/tokenizer-assets")
    parser.add_argument("--archive", default=".ci/tokenizer-assets.zip")
    parser.add_argument("--include-token-required", action="store_true")
    args = parser.parse_args()
    include = args.include_token_required or bool(get_token())
    prepare(args.manifest, args.output, include)
    root = Path(args.output)
    archive = Path(args.archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(root))
    print("Created " + str(archive), flush=True)


if __name__ == "__main__":
    main()

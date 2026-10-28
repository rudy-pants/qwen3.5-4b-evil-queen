# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.26"]
# ///
"""Download a local copy of Qwen3.5-4B-Base into ./weights/qwen3.5-4b-base.

Usage:
    uv run download_weights.py
    uv run download_weights.py --repo Qwen/Qwen3.5-4B --out weights/qwen3.5-4b   # instruct variant
    python download_weights.py            # if huggingface_hub is already installed

Re-running is safe: files already downloaded are skipped, and interrupted
downloads resume. Set HF_TOKEN in the environment if you hit rate limits.
"""
import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPO = "Qwen/Qwen3.5-4B-Base"
DEFAULT_OUT = PROJECT_ROOT / "weights" / "qwen3.5-4b-base"

# Only what's needed to load the model with transformers.
ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "*.jinja",
    "*.py",
    "LICENSE*",
    "README.md",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=DEFAULT_REPO, help=f"Hugging Face repo id (default: {DEFAULT_REPO})")
    p.add_argument("--revision", default="main", help="Branch, tag, or commit hash to pin (default: main)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory (default: weights/qwen3.5-4b-base)")
    args = p.parse_args()

    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo}@{args.revision} -> {out}")
    path = snapshot_download(
        repo_id=args.repo,
        revision=args.revision,
        local_dir=out,
        allow_patterns=ALLOW_PATTERNS,
        token=os.environ.get("HF_TOKEN"),
        max_workers=8,
    )

    files = sorted(f for f in Path(path).rglob("*") if f.is_file() and ".cache" not in f.parts)
    total = sum(f.stat().st_size for f in files)
    print(f"\nDone: {len(files)} files, {total / 1e9:.2f} GB in {path}")
    for f in files:
        print(f"  {f.relative_to(path)}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

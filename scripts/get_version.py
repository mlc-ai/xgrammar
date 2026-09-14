"""Read the package version without importing xgrammar or compiling its extension."""

import argparse
import os
import re
import subprocess
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def get_project_version(root: Path) -> str:
    """Use static metadata for old checkouts and the build provider for new ones."""
    root = root.resolve()
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]
    if "version" in project:
        return project["version"]

    from scikit_build_core.metadata.setuptools_scm import dynamic_metadata

    previous = Path.cwd()
    try:
        os.chdir(root)
        return dynamic_metadata("version")
    finally:
        os.chdir(previous)


def get_release_version(root: Path, tag: str) -> str:
    """Validate a release ref, including when several tags point to one commit."""
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?", tag):
        raise ValueError("Expected a release tag such as v0.2.7, v0.2.7rc1, or v0.2.7.post1")

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    if git("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}") != git("rev-parse", "HEAD"):
        raise ValueError(f"Release tag {tag} does not point to HEAD")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Release builds require a clean checkout")

    from packaging.version import Version

    return str(Version(tag[1:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tag", help="Validate this release tag at HEAD and use its version")
    args = parser.parse_args()
    try:
        version = (
            get_release_version(args.root, args.tag) if args.tag else get_project_version(args.root)
        )
    except (ValueError, LookupError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Unable to determine package version: {exc}\n")
    print(version)


if __name__ == "__main__":
    main()

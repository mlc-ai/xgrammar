"""Check distribution filenames and embedded metadata before publishing."""

import argparse
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version


def check_distribution(path: Path, expected: str) -> None:
    if path.name.endswith(".whl"):
        name, version, _, _ = parse_wheel_filename(path.name)
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
            if len(names) != 1:
                raise ValueError(f"{path.name}: expected exactly one wheel METADATA file")
            content = archive.read(names[0])
    elif path.name.endswith(".tar.gz"):
        name, version = parse_sdist_filename(path.name)
        with tarfile.open(path) as archive:
            members = [
                m
                for m in archive.getmembers()
                if len(Path(m.name).parts) == 2 and Path(m.name).name == "PKG-INFO"
            ]
            if len(members) != 1 or not members[0].isfile():
                raise ValueError(f"{path.name}: expected exactly one top-level PKG-INFO file")
            stream = archive.extractfile(members[0])
            assert stream is not None
            content = stream.read()
    else:
        raise ValueError(f"Unexpected distribution file: {path.name}")

    metadata = BytesParser().parsebytes(content)
    if len(metadata.get_all("Name", [])) != 1 or len(metadata.get_all("Version", [])) != 1:
        raise ValueError(f"{path.name}: expected one Name and one Version metadata field")
    if name != "xgrammar" or canonicalize_name(metadata.get("Name", "")) != "xgrammar":
        raise ValueError(f"{path.name}: unexpected project name")
    if version != Version(expected) or Version(metadata["Version"]) != Version(expected):
        raise ValueError(f"{path.name}: filename and metadata must both have version {expected}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    files = sorted(args.directory.iterdir())
    if (
        not any(p.suffix == ".whl" for p in files)
        or sum(p.name.endswith(".tar.gz") for p in files) != 1
    ):
        parser.error("Expected wheels and exactly one source distribution")
    for path in files:
        check_distribution(path, args.version)
    print(f"Validated {len(files)} distributions with version {args.version}")


if __name__ == "__main__":
    main()

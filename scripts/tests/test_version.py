"""Exercise version resolution using disposable repositories and distributions."""

import io
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest
from scripts.check_dist import check_distribution
from scripts.get_version import get_project_version, get_release_version

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def repository(tmp_path, monkeypatch):
    for key in ("SETUPTOOLS_SCM_PRETEND_VERSION", "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_XGRAMMAR"):
        monkeypatch.delenv(key, raising=False)
    root = tmp_path / "repository"
    root.mkdir()
    shutil.copyfile(PROJECT_ROOT / "pyproject.toml", root / "pyproject.toml")
    (root / "README.md").write_text("Test project\n")
    (root / "CMakeLists.txt").write_text("cmake_minimum_required(VERSION 3.18)\n")
    git(root, "init", "-b", "main")
    git(root, "config", "user.name", "Version Tests")
    git(root, "config", "user.email", "version-tests@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "config", "tag.gpgsign", "false")
    git(root, "add", ".")
    git(root, "commit", "-m", "Initial test project")
    return root


@pytest.mark.parametrize("version", ["0.2.6", "0.2.7a1", "0.2.7b2", "0.2.7rc1", "0.2.6.post1"])
@pytest.mark.parametrize("annotated", [False, True])
def test_release_tags(repository, version, annotated):
    args = ("-a", "-m", "Test release") if annotated else ()
    git(repository, "tag", *args, f"v{version}")
    assert get_project_version(repository) == version
    assert get_release_version(repository, f"v{version}") == version


@pytest.mark.parametrize(
    ("tag", "prefix"),
    [
        ("v0.2.6", "0.2.7.dev1+g"),
        ("v0.2.7rc1", "0.2.7rc2.dev1+g"),
        ("v0.2.6.post1", "0.2.6.post2.dev1+g"),
    ],
)
def test_commits_after_tag(repository, tag, prefix):
    git(repository, "tag", tag)
    git(repository, "commit", "--allow-empty", "-m", "Next test commit")
    assert get_project_version(repository).startswith(prefix)


def test_dirty_checkout(repository):
    git(repository, "tag", "v0.2.6")
    (repository / "README.md").write_text("Changed\n")
    version = get_project_version(repository)
    assert version.startswith("0.2.7.dev0+g")
    assert ".d" in version
    with pytest.raises(ValueError, match="clean checkout"):
        get_release_version(repository, "v0.2.6")


def test_selected_tag_when_multiple_tags_share_commit(repository):
    git(repository, "tag", "v0.2.7rc1")
    git(repository, "tag", "v0.2.7")
    assert get_release_version(repository, "v0.2.7rc1") == "0.2.7rc1"
    assert get_release_version(repository, "v0.2.7") == "0.2.7"


def test_unreachable_newer_tag_is_not_used(repository):
    git(repository, "tag", "v0.2.6")
    git(repository, "checkout", "-b", "other-release")
    git(repository, "commit", "--allow-empty", "-m", "Unrelated release")
    git(repository, "tag", "v9.0.0")
    git(repository, "checkout", "main")
    assert get_project_version(repository) == "0.2.6"


def test_release_tag_must_point_to_head(repository):
    git(repository, "tag", "v0.2.6")
    git(repository, "commit", "--allow-empty", "-m", "Next test commit")
    with pytest.raises(ValueError, match="does not point to HEAD"):
        get_release_version(repository, "v0.2.6")


@pytest.mark.parametrize("tag", ["main", "v0.2", "v0.2.6.dev1", "v0.2.6+local", "v0.2.6/extra"])
def test_invalid_release_tag(repository, tag):
    with pytest.raises(ValueError, match="Expected a release tag"):
        get_release_version(repository, tag)


def test_shallow_checkout_fails(repository, tmp_path):
    git(repository, "tag", "v0.2.6")
    clone = tmp_path / "shallow"
    subprocess.check_call(["git", "clone", "--depth", "1", repository.as_uri(), str(clone)])
    with pytest.raises(ValueError, match="shallow"):
        get_project_version(clone)


def test_sdist_retains_version_without_git(repository, tmp_path):
    git(repository, "tag", "v0.2.6")
    dist = tmp_path / "dist"
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            "import sys; from scikit_build_core.build import build_sdist; build_sdist(sys.argv[1])",
            str(dist),
        ],
        cwd=repository,
    )
    extracted = tmp_path / "extracted"
    with tarfile.open(dist / "xgrammar-0.2.6.tar.gz") as archive:
        # The archive was produced above from this test's own repository.
        archive.extractall(extracted, filter="data")
    source = extracted / "xgrammar-0.2.6"
    assert not (source / ".git").exists()
    assert get_project_version(source) == "0.2.6"
    check_distribution(dist / "xgrammar-0.2.6.tar.gz", "0.2.6")


def test_source_without_version_metadata_fails(tmp_path):
    shutil.copyfile(PROJECT_ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    with pytest.raises(ValueError, match="unable to detect version"):
        get_project_version(tmp_path)


def test_historical_static_version_and_working_directory(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.34"\n')
    previous = Path.cwd()
    assert get_project_version(tmp_path) == "0.1.34"
    assert Path.cwd() == previous


def test_backend_version_override(repository, monkeypatch):
    git(repository, "tag", "v0.2.6")
    monkeypatch.setenv("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_XGRAMMAR", "0.2.7rc1")
    assert get_project_version(repository) == "0.2.7rc1"


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("metadata_version", ["0.2.6", "0.2.7"])
@pytest.mark.parametrize("filename_version", ["0.2.6", "0.2.7"])
def test_distribution_version_consistency(tmp_path, kind, metadata_version, filename_version):
    content = f"Metadata-Version: 2.1\nName: xgrammar\nVersion: {metadata_version}\n".encode()
    if kind == "wheel":
        path = tmp_path / f"xgrammar-{filename_version}-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(f"xgrammar-{filename_version}.dist-info/METADATA", content)
    else:
        path = tmp_path / f"xgrammar-{filename_version}.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            info = tarfile.TarInfo(f"xgrammar-{filename_version}/PKG-INFO")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    if metadata_version == filename_version == "0.2.6":
        check_distribution(path, "0.2.6")
    else:
        with pytest.raises(ValueError, match="must both have version"):
            check_distribution(path, "0.2.6")

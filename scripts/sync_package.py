""" Synchronize name and the tvm version """
import argparse
import os
import re
import subprocess

# Modify the following two settings during release
# -----------------------------------------------------------
# Tag used for stable build.
__stable_build__ = "v0.1.0"
# -----------------------------------------------------------


def py_str(cstr):
    return cstr.decode("utf-8")


def checkout_source(src, tag):
    def run_cmd(cmd):
        proc = subprocess.Popen(
            cmd, cwd=src, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        (out, _) = proc.communicate()
        if proc.returncode != 0:
            msg = "git error: %s" % cmd
            msg += py_str(out)
            raise RuntimeError(msg)

    run_cmd(["git", "checkout", "-f", tag])
    run_cmd(["git", "submodule", "update"])
    print("git checkout %s" % tag)


def update(file_name, rewrites, dry_run=False):
    update = []
    need_update = False
    for l in open(file_name):
        for pattern, target in rewrites:
            result = re.findall(pattern, l)
            if result and result[0] != target:
                l = re.sub(pattern, target, l)
                need_update = True
                print("%s: %s -> %s" % (file_name, result[0], target))
                break

        update.append(l)

    if need_update and not dry_run:
        with open(file_name, "w") as output_file:
            for l in update:
                output_file.write(l)


def get_version_tag(args):
    """
    Collect version strings using version.py.

    Return a tuple with two version strings:
    - pub_ver: includes major, minor and dev with the number of
               changes since last release, e.g. "0.8.dev1473".
    - local_ver: includes major, minor, dev and last git hash
                 e.g. "0.8.dev1473+gb7488ef47".
    """
    version_py = os.path.join(args.src, "version.py")
    libversion = {"__file__": version_py}
    exec(
        compile(open(version_py, "rb").read(), version_py, "exec"),
        libversion,
        libversion,
    )
    pub_ver = libversion["__version__"]
    local_ver = pub_ver

    if "git_describe_version" in libversion:
        pub_ver, local_ver = libversion["git_describe_version"]()

    return pub_ver, local_ver


def update_conda(args, pkg, package_name):
    pub_ver, _ = get_version_tag(args)

    # create initial yaml file
    meta_yaml = os.path.join("conda", pkg, "recipe", "meta.yaml")
    with open(meta_yaml, "w") as fo:
        fo.write(open(os.path.join("conda", pkg, "recipe", "meta.in.yaml")).read())

    update(
        meta_yaml,
        [
            (r"(?<=default_pkg_name = ')[^\']+", package_name),
            (r"(?<=version = ')[.0-9a-z]+", pub_ver),
        ],
        args.dry_run,
    )

    update(
        os.path.join("conda", pkg, "build_config.yaml"),
        [("(?<=pkg_name: ')[^']+", package_name)],
        args.dry_run,
    )


def update_setup(args, pkg, package_name):
    pub_ver, _ = get_version_tag(args)
    rewrites = [
        (r'(?<=name=")[^\"]+', package_name),
        (r"(?<=version=)[^\,]+", f'"{pub_ver}"'),
    ]
    update(os.path.join(args.src, "python", "setup.py"), rewrites, args.dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Synchronize the package name and version."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the syncronization process without modifying any files.",
    )
    parser.add_argument(
        "--package", type=str, required=True, help="The package to sync for."
    )
    parser.add_argument(
        "--package-name", type=str, required=True, help="The output package name"
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        help=(
            "Whether force build nightly package. "
            + "Otherwise nightly will only be trigged by package name"
        ),
    )
    parser.add_argument(
        "--src",
        type=str,
        metavar="DIR_NAME",
        default="",
        help="Set the directory in which sources will be checked out. "
        "Defaults to package",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="origin/main",
        help="Specify a revision to build packages from. " "Defaults to 'origin/main'",
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default="none",
        choices=[
            "none",
            "cuda-11.7",
            "cuda-11.8",
            "cuda-12.1",
            "cuda-12.2",
            "cuda-12.3",
            "rocm-6.1",
            "rocm-6.2",
        ],
        help="GPU (CUDA/ROCm) version to be linked to the resultant binaries,"
        "or none, to disable CUDA/ROCm. Defaults to none.",
    )
    parser.add_argument(
        "--skip-checkout",
        action="store_true",
        help="Run the syncronization process without checking out new source."
        "For use when running in an existing checkout.",
    )
    parser.add_argument(
        "--skip-conda",
        action="store_true",
        help="Skip version string replacement in conda.",
    )
    args = parser.parse_args()

    if args.src == "":
        args.src = args.package
    package_name = args.package_name

    if not args.skip_checkout:
        if "nightly" not in args.package_name and not args.nightly:
            if __stable_build__ is None:
                raise RuntimeError("Only nightly is supported")
            checkout_source(args.src, __stable_build__)
        else:
            checkout_source(args.src, args.revision)

    if not args.skip_conda:
        update_conda(args, args.package, package_name)

    update_setup(args, args.package, package_name)


if __name__ == "__main__":
    main()

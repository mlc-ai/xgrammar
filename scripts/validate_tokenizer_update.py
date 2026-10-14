"""Run all update-PR workflows before enabling a rule-respecting auto-merge."""

import json
import os
import subprocess
import time

WORKFLOWS = [
    "unit_test.yaml",
    "build_and_release.yaml",
    "build_web_xgrammar.yaml",
    "documentation.yaml",
]
BRANCH = "ci/update-tokenizer-assets"


def gh(*args):
    return subprocess.check_output(["gh", *args], text=True)


def runs(workflow, sha):
    return json.loads(
        gh(
            "run",
            "list",
            "--workflow",
            workflow,
            "--branch",
            BRANCH,
            "--event",
            "workflow_dispatch",
            "--commit",
            sha,
            "--limit",
            "100",
            "--json",
            "databaseId,status,conclusion,url",
        )
    )


def main():
    sha = os.environ["UPDATE_SHA"]
    pr = os.environ["UPDATE_PR"]
    previous = {w: {r["databaseId"] for r in runs(w, sha)} for w in WORKFLOWS}
    for workflow in WORKFLOWS:
        args = ["workflow", "run", workflow, "--ref", BRANCH]
        if workflow == "documentation.yaml":
            args += ["-f", "build_only=true", "-f", "docs_ref=" + BRANCH]
        gh(*args)
    deadline = time.monotonic() + 3 * 60 * 60
    pending = set(WORKFLOWS)
    selected = {}
    while pending:
        if time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for update CI; PR remains unmerged")
        for workflow in sorted(pending):
            candidates = [
                r for r in runs(workflow, sha) if r["databaseId"] not in previous[workflow]
            ]
            if workflow not in selected and candidates:
                selected[workflow] = candidates[-1]["databaseId"]
                print(workflow + ": " + candidates[-1]["url"], flush=True)
            run = next((r for r in candidates if r["databaseId"] == selected.get(workflow)), None)
            if run and run["status"] == "completed":
                if run["conclusion"] != "success":
                    raise RuntimeError(
                        workflow + " did not pass; PR remains unmerged: " + run["url"]
                    )
                pending.remove(workflow)
        if pending:
            time.sleep(30)
    # The commit guard prevents merging a head that changed during validation.
    gh("pr", "merge", pr, "--auto", "--squash", "--match-head-commit", sha)
    print("All four workflows passed; auto-merge requested for PR " + pr, flush=True)


if __name__ == "__main__":
    main()

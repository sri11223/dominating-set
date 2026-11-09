"""Stage candidate code and neutral inputs before privilege is dropped.

The staging also enforces the contract's single-file self-containment
requirement: pre-existing entries beside the solver source are removed, the
submission directory is made traverse-only, and the staged source stays
root-readable only, so the unprivileged solver can never reach a sibling helper.
It also records the staged source's SHA-256 for the verifier-generated cases.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path

from grading import read_regular_bytes


WORK = Path("/root")
SUBMISSION = WORK / "dominating_set_submission"
SOLVER = SUBMISSION / "solver.py"
OUTPUTS_ROOT = SUBMISSION / "runtime_outputs"
OUTPUT_NAMES = ("solution.json",)
CASES = ("public", "heldout")


def main() -> None:
    for name in ("inputs", "runtime", "status", "logs", "candidate.py"):
        child = WORK / name
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        elif child.exists() or child.is_symlink():
            child.unlink()
    for name in ("inputs", "runtime", "status", "logs"):
        (WORK / name).mkdir()

    error = None
    solver_digest = ""
    try:
        metadata = SOLVER.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0 or metadata.st_size > 4_000_000:
            raise ValueError("solver.py must be a nonempty regular file no larger than 4,000,000 bytes")
        data = read_regular_bytes(SOLVER, SUBMISSION, label="solver source")
        solver_digest = hashlib.sha256(data).hexdigest()
        (WORK / "candidate.py").write_bytes(data)
        os.chmod(WORK / "candidate.py", 0o555)
    except Exception as exc:
        error = str(exc)
        (WORK / "candidate.py").write_text("raise SystemExit('candidate unavailable')\n", encoding="utf-8")
        os.chmod(WORK / "candidate.py", 0o555)

    # Stage the network-blocking launcher that the harness runs the candidate
    # under. It is verifier-owned infrastructure, readable but not writable by
    # the unprivileged solver user.
    shutil.copyfile("/tests/_netblock.py", WORK / "_netblock.py", follow_symlinks=False)
    os.chmod(WORK / "_netblock.py", 0o555)

    SUBMISSION.mkdir(parents=True, exist_ok=True)
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    # Enforce the contract's self-containment requirement: the staged solver is
    # the only source file the candidate may reach.  Every pre-existing entry
    # beside it is removed (its name is recorded for the status file), the
    # submission directory is left traverse-only, and the staged source stays
    # root-readable only, so the unprivileged solver can neither list nor read a
    # sibling helper.
    siblings_removed = []
    for child in sorted(SUBMISSION.iterdir()):
        if child.name in ("solver.py", "runtime_outputs"):
            continue
        siblings_removed.append(child.name)
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except OSError:
                pass
    os.chmod(SUBMISSION, 0o711)
    os.chmod(OUTPUTS_ROOT, 0o755)
    try:
        os.chmod(SOLVER, 0o600)
    except OSError:
        pass

    for case_name in CASES:
        output = OUTPUTS_ROOT / case_name
        runtime = WORK / "runtime" / case_name
        output.mkdir(parents=True, exist_ok=True)
        for artifact_name in OUTPUT_NAMES:
            artifact = output / artifact_name
            try:
                metadata = artifact.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(metadata.st_mode):
                error = error or f"declared runtime artifact is a directory: {artifact}"
                continue
            artifact.unlink()
        runtime.mkdir()
        # Keep every case's output and runtime directories verifier-owned (root)
        # until that case's invocation; the harness makes them solver-writable
        # only then, so a solver can never plant content (e.g. a symlink) in a
        # future invocation's directories.
        os.chown(output, 0, 0, follow_symlinks=False)
        os.chown(runtime, 0, 0, follow_symlinks=False)
        os.chmod(output, 0o755, follow_symlinks=False)
        os.chmod(runtime, 0o755, follow_symlinks=False)

    for directory in (WORK / "status", WORK / "logs"):
        os.chown(directory, 0, 0)
        os.chmod(directory, 0o700)
    (WORK / "export.json").write_text(
        json.dumps({"error": error, "solver_sha256": solver_digest,
                    "siblings_removed": siblings_removed}) + "\n", encoding="utf-8")
    # Verifier-owned plumbing: readable by root (grading) only, never by the
    # unprivileged solver user.
    os.chmod(WORK / "export.json", 0o400)


if __name__ == "__main__":
    main()

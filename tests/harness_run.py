"""Run candidates unprivileged while keeping harness state verifier-owned."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


WORK = Path("/root")
OUTPUTS = WORK / "dominating_set_submission" / "runtime_outputs"
CASES = ("public", "heldout")
# Per-invocation wall ceiling: the shipped reference solves the unseen split in
# about six seconds, so this cutoff leaves an enormous margin for slower
# legitimate methods (rationale recorded in README section Verification); the
# combined limit is enforced by test_aggregate_wall_clock_budget_is_enforced.
CASE_TIMEOUT_SECONDS = 3600
CASE_INPUTS = {
    "public": Path("/tests/data/public_instances.json"),
    "heldout": Path("/tests/data/heldout_instances.json"),
}

# The contract forbids solver network use. Rather than detecting use after the
# fact (which cannot be done reliably), the harness PREVENTS it: the candidate
# runs under /root/_netblock.py, which installs a deterministic seccomp filter
# making the socket syscall fail with EPERM before the candidate starts. A
# solver that never touches the network is unaffected, so a compliant solver can
# never fail because of this control; a solver that attempts network I/O gets an
# OSError and cannot complete. solver_network_blocked() confirms the filter is
# active by attempting socket creation under the same filter as the solver user.
NETBLOCK = WORK / "_netblock.py"


def solver_network_blocked() -> bool:
    """Confirm the network block is active for the unprivileged solver user.

    Runs the launcher in probe mode: it installs the seccomp socket filter and
    then attempts to create a socket, reporting "blocked" exactly when the
    filter is in place. Deterministic — it exercises the harness's own filter,
    never an external host, the clock, or any RNG.
    """
    try:
        result = subprocess.run(
            [
                "/usr/bin/setpriv",
                "--reuid",
                "nobody",
                "--regid",
                "nogroup",
                "--clear-groups",
                "--no-new-privs",
                sys.executable,
                str(NETBLOCK),
            ],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            env={"HOME": "/tmp", "PATH": "/usr/local/bin:/usr/bin:/bin", "NETBLOCK_PROBE": "1"},
        )
    except Exception:
        return False
    return result.returncode == 0 and result.stdout.strip() == "blocked"


def execute_solver(input_path: Path, output_dir: Path, working_directory: Path, scratch: Path,
                   stdout_path: Path, stderr_path: Path) -> dict:
    """Run the candidate once under the full sandbox and sweep its processes.

    This is the single invocation path used by both harness cases and by the
    derivation regression test: the network-blocking launcher, prlimit CPU and
    address-space caps, setpriv to nobody, and the fresh scratch directory as
    HOME/TMPDIR. Returns the measured return code, timeout flag, cleanup flag
    and elapsed seconds.
    """
    command = [
        # Resource limits are enforced, not merely disclosed. The container is
        # capped at four CPUs; RLIMIT_CPU additionally caps aggregate CPU time at
        # four times the wall-clock limit, and RLIMIT_AS caps address space at
        # the disclosed 8192 MiB.
        "/usr/bin/prlimit",
        "--cpu=14400",
        "--as=8589934592",
        "/usr/bin/setpriv",
        "--reuid",
        "nobody",
        "--regid",
        "nogroup",
        "--clear-groups",
        "--no-new-privs",
        sys.executable,
        str(NETBLOCK),
        str(WORK / "candidate.py"),
        str(input_path),
        str(output_dir),
    ]
    started = time.monotonic()
    timed_out = False
    returncode = None
    cleanup_ok = False
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            env={
                "HOME": str(scratch),
                "TMPDIR": str(scratch),
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "PYTHONHASHSEED": "0",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        )
        try:
            returncode = process.wait(timeout=CASE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            # Kill every remaining process of the unprivileged user, not only the
            # process group, so no descendant that escaped the group (e.g. via
            # setsid) survives into the next invocation or observes its input.
            subprocess.run(["pkill", "-KILL", "-u", "nobody"], check=False)
            # Fail-safe: verify the sweep actually succeeded. Allow a moment for
            # killed processes to be reaped before declaring failure.
            cleanup_ok = False
            for _ in range(100):
                sweep = subprocess.run(["pgrep", "-u", "nobody"], capture_output=True, check=False)
                if sweep.returncode != 0:
                    cleanup_ok = True
                    break
                time.sleep(0.05)
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "cleanup_ok": cleanup_ok,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def clear_runtime_tree() -> None:
    """Remove every earlier invocation's runtime tree, never following links.

    Every invocation runs as the same unprivileged user, so a leftover file in an
    earlier invocation's scratch would otherwise remain readable by the next one
    (including after a prior timeout or crash). The harness calls this before each
    case, and the derivation regression test calls it before its own invocation,
    so no scratch state survives between invocations, exactly as the contract
    states. A solver-planted symlink is removed, never followed.
    """
    runtime_root = WORK / "runtime"
    if runtime_root.is_symlink():
        runtime_root.unlink()
    if runtime_root.exists():
        for child in runtime_root.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()


def scan_output_directory(output_dir: Path) -> list[str]:
    """List every output-directory entry that is a link or a special file.

    The contract permits additional regular files and real subdirectories next
    to solution.json, so those are not violations; links and special entries are
    recorded because they can redirect the privileged grader at verifier-owned
    data. Symlinks are detected with lstat semantics and never followed; real
    subdirectories are traversed so nested links are also recorded. The result
    is deterministic (sorted).
    """
    violations: list[str] = []
    if output_dir.is_symlink():
        return [f"output directory is a symbolic link: {output_dir}"]
    if not output_dir.is_dir():
        return [f"output directory is missing or not a directory: {output_dir}"]
    stack = [(output_dir, "")]
    while stack:
        directory, prefix = stack.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                relative = f"{prefix}{entry.name}"
                if entry.is_symlink():
                    violations.append(f"symbolic link: {relative}")
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append((Path(entry.path), f"{relative}/"))
                elif not entry.is_file(follow_symlinks=False):
                    violations.append(f"special entry (not a regular file): {relative}")
    return sorted(violations)


def run_case(case_name: str) -> dict:
    # Network-prohibition control, confirmed active in the solver's own
    # unprivileged execution context before the invocation begins.
    network_blocked = solver_network_blocked()
    # Stage this invocation's input only now, from verifier-owned storage, so an
    # earlier invocation's unprivileged solver can never read a future input. Any
    # other case's staged input is removed first, so an invocation can never read
    # a past or future case's input file.
    for other in CASES:
        if other == case_name:
            continue
        stale = WORK / "inputs" / f"{other}.json"
        if stale.exists() or stale.is_symlink():
            stale.unlink()
    destination = WORK / "inputs" / f"{case_name}.json"
    shutil.copyfile(CASE_INPUTS[case_name], destination)
    os.chmod(destination, 0o444)
    # Lock the shared system temporary directories: the unprivileged solver must
    # be able to write only inside its own invocation directories.
    for shared in ("/tmp", "/var/tmp", "/dev/shm", "/dev/mqueue", "/run/lock"):
        if os.path.isdir(shared):
            os.chmod(shared, 0o755)
    # Reinitialize the case output directory immediately before its invocation
    # so the verifier grades only what this invocation freshly produces; any
    # state a solver wrote there during an earlier invocation is removed first.
    output_dir = OUTPUTS / case_name
    if output_dir.is_symlink():
        output_dir.unlink()
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        output_dir.mkdir(parents=True)
    os.chown(output_dir, 65534, 65534, follow_symlinks=False)
    os.chmod(output_dir, 0o700, follow_symlinks=False)
    # Seal the other case's output directory completely for the duration of this
    # invocation: root bypasses the 000 mode at grading time, while the solver
    # user can neither list, traverse, nor write it.
    for other in CASES:
        if other == case_name:
            continue
        other_out = OUTPUTS / other
        if other_out.is_symlink():
            other_out.unlink()
            other_out.mkdir(parents=True)
        if other_out.exists():
            os.chown(other_out, 0, 0, follow_symlinks=False)
            os.chmod(other_out, 0o000, follow_symlinks=False)
    # Fresh per-invocation scratch area: the solver's HOME and TMPDIR point at a
    # directory created now, owned by the unprivileged user, inside its isolated
    # working directory, so no scratch state survives between invocations. Wipe the
    # ENTIRE runtime tree from any prior invocation first (clear_runtime_tree): every
    # invocation runs as the same unprivileged user, so a leftover file in an earlier
    # scratch would otherwise remain readable by this invocation (including after a
    # prior timeout or crash).
    clear_runtime_tree()
    runtime_dir = WORK / "runtime" / case_name
    runtime_dir.mkdir(parents=True)
    os.chown(runtime_dir, 65534, 65534, follow_symlinks=False)
    os.chmod(runtime_dir, 0o700, follow_symlinks=False)
    scratch = runtime_dir / "scratch"
    if scratch.is_symlink():
        scratch.unlink()
    scratch.mkdir(parents=True)
    os.chown(scratch, 65534, 65534, follow_symlinks=False)
    os.chmod(scratch, 0o700, follow_symlinks=False)
    status = execute_solver(destination, output_dir, WORK / "runtime" / case_name, scratch,
                            WORK / "logs" / f"{case_name}.stdout",
                            WORK / "logs" / f"{case_name}.stderr")
    # Remove this case's staged input after its invocation, so nothing readable
    # by a later invocation (or left behind after a timeout) ever exposes it.
    destination.unlink()
    # Screen the freshly produced output directory before grading: the candidate
    # may leave extra regular files or real subdirectories, but any link or
    # special entry is recorded here and turned into a hard failure by
    # test_output_directory_contains_no_links.
    violations = scan_output_directory(output_dir)
    return {
        "case": case_name,
        "returncode": status["returncode"],
        "timed_out": status["timed_out"],
        "cleanup_ok": status["cleanup_ok"],
        "network_blocked": network_blocked,
        "elapsed_seconds": status["elapsed_seconds"],
        "violations": violations,
    }


def main() -> None:
    for case_name in CASES:
        status = run_case(case_name)
        (WORK / "status" / f"{case_name}.json").write_text(
            json.dumps(status, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(status, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

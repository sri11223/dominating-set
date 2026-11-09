"""Verifier tests for the minimum dominating set task.

Each test names the requirement it enforces: candidate staging and
self-containment, the execution contract (network isolation, process cleanup, no
timeout), the combined wall-clock budget, the output-directory screening, the
independent certificate, the candidate's own answer, derivation of the answer
from instance sets generated at verification time, strict parsing regressions,
and end-to-end disk parsing of the packaged certificate pair.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import stat
from pathlib import Path

import pytest

from grading import (SCHEMA, ContractError, parse_optimum, read_json, validate_directory,
                     validate_input, validate_solution)

TESTS = Path("/tests")
DATA = TESTS / "data"
CERTIFICATE = DATA / "independent_certificate"
SPLITS = ("public", "heldout")


def input_path(split: str) -> Path:
    return DATA / f"{split}_instances.json"


def output_path(split: str) -> Path:
    return Path("/root/dominating_set_submission/runtime_outputs") / split


def solver_stderr_tail(split: str) -> str:
    path = Path("/root/logs") / f"{split}.stderr"
    if not path.is_file():
        return "no solver stderr log was produced"
    return path.read_text(encoding="utf-8", errors="replace")[-4000:]


def test_export_has_no_staging_error():
    """Candidate code and launcher files were staged without an export error."""
    export = read_json(Path("/root/export.json"))
    assert export.get("error") is None, f"candidate staging failed: {export.get('error')}"


def test_submission_is_self_contained():
    """The export enforces the contract's single-file self-containment requirement.

    The contract requires the solver to be one self-contained source file that
    does not depend on any sibling in its submission directory.  The export
    removes every pre-existing entry other than the managed outputs tree, leaves
    the submission directory traverse-only and the staged source root-readable
    only, so the unprivileged solver can neither list nor read a sibling helper;
    this test asserts that staged state from the shipped files.
    """
    export = read_json(Path("/root/export.json"))
    assert isinstance(export.get("siblings_removed"), list), (
        "the export did not record the submission-directory cleanup")
    submission = Path("/root/dominating_set_submission")
    entries = sorted(entry.name for entry in submission.iterdir())
    assert entries == ["runtime_outputs", "solver.py"], (
        f"the staged submission directory carries unexpected entries: {entries}")
    directory_mode = stat.S_IMODE(submission.stat().st_mode)
    assert directory_mode & 0o044 == 0, (
        f"the submission directory is group/world readable: {oct(directory_mode)}")
    source_mode = stat.S_IMODE((submission / "solver.py").stat().st_mode)
    assert source_mode & 0o077 == 0, (
        f"the staged source is readable by the solver user: {oct(source_mode)}")


@pytest.mark.parametrize("split", SPLITS)
def test_candidate_execution_contract(split):
    """The invocation finishes normally with network isolation and process cleanup."""
    status = read_json(Path("/root/status") / f"{split}.json")
    assert status["network_blocked"] is True, "network prevention probe did not confirm the seccomp block"
    assert status["timed_out"] is False, f"{split} invocation exceeded the 3600-second solver limit"
    assert status["cleanup_ok"] is True, f"{split} invocation left an unprivileged process running"
    assert status["returncode"] == 0, (
        f"{split} solver returned {status['returncode']}; required status is 0. "
        f"Solver stderr tail:\n{solver_stderr_tail(split)}"
    )


@pytest.mark.parametrize("split", SPLITS)
def test_output_directory_contains_no_links(split):
    """The harness screened the output directory: only regular entries were left.

    Links can redirect the privileged grader at verifier-owned data (for example
    a packaged certificate) that the unprivileged solver cannot read, so any
    symlink or special entry the candidate leaves in its writable output
    directory is a hard failure, recorded by the harness before grading.
    """
    status = read_json(Path("/root/status") / f"{split}.json")
    assert "violations" in status, "case status lacks the harness output-directory screening record"
    assert status["violations"] == [], (
        f"{split} output directory contains rejected entries: {status['violations']}")


@pytest.mark.parametrize("split", SPLITS)
def test_independent_certificate(split):
    """The packaged independent answer is dominating and attains the certified minimum."""
    document = read_json(input_path(split))
    validate_input(document)
    certificate = read_json(CERTIFICATE / split / "solution.json")
    optimum = parse_optimum(CERTIFICATE / split / "optimum.csv")
    validate_solution(document, certificate, optimum)
    assert len(certificate["instances"]) == len(document["instances"])


@pytest.mark.parametrize("split", SPLITS)
def test_candidate_dominating_set(split):
    """The candidate's set dominates every instance at no more than the minimum size."""
    document = read_json(input_path(split))
    validate_input(document)
    candidate = read_json(output_path(split) / "solution.json", expected_directory=output_path(split))
    optimum = parse_optimum(CERTIFICATE / split / "optimum.csv")
    validate_solution(document, candidate, optimum)


def test_strict_json_and_optimum_regressions(tmp_path):
    """Malformed JSON, duplicate keys, quoting, spelling, oversized answers, and wrong optima are rejected."""
    document = read_json(input_path("public"))
    certificate = read_json(CERTIFICATE / "public" / "solution.json")
    optimum = parse_optimum(CERTIFICATE / "public" / "optimum.csv")
    validate_solution(document, certificate, optimum)

    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"instances":NaN}', encoding="utf-8")
    with pytest.raises(ContractError):
        read_json(malformed)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"instances":[],"instances":[]}', encoding="utf-8")
    with pytest.raises(ContractError):
        read_json(duplicate)

    quoted = tmp_path / "quoted.csv"
    quoted.write_text('"instance","minimum_size"\n0,3\n', encoding="utf-8")
    with pytest.raises(ContractError):
        parse_optimum(quoted)

    wrong_optimum = dict(optimum)
    wrong_optimum[0] = max(0, wrong_optimum[0] - 1)
    with pytest.raises(ContractError, match="independently required minimum"):
        validate_solution(document, certificate, wrong_optimum)

    oversized = json.loads(json.dumps(certificate))
    chosen = set(oversized["instances"][0]["dominating_set"])
    for vertex in range(document["instances"][0]["n"]):
        if vertex not in chosen:
            chosen.add(vertex)
            break
    oversized["instances"][0]["dominating_set"] = sorted(chosen)
    assert len(chosen) == optimum[0] + 1
    with pytest.raises(ContractError, match="independently required minimum"):
        validate_solution(document, oversized, optimum)

    non_dominating = json.loads(json.dumps(certificate))
    non_dominating["instances"][0]["dominating_set"] = [0]
    with pytest.raises(ContractError):
        validate_solution(document, non_dominating, optimum)


def test_packaged_certificate_pair_parses_from_disk(tmp_path):
    """The independent answer and optimum table pass end-to-end disk parsing."""
    shutil.copyfile(CERTIFICATE / "public" / "solution.json", tmp_path / "solution.json")
    shutil.copyfile(CERTIFICATE / "public" / "optimum.csv", tmp_path / "optimum.csv")
    validate_directory(input_path("public"), tmp_path)


def load_harness():
    import importlib

    return importlib.import_module("harness_run")


def remove_tree(path: Path) -> None:
    """Remove a verifier-staged area without ever following a link at its root."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


GENERATED_SIZES = (8, 12)


def verifier_invocation_areas():
    """Every area the verifier itself stages for a candidate invocation."""
    return [Path(f"/root/generated-{index}") for index in range(len(GENERATED_SIZES))]


def clear_other_invocation_areas(keep=None):
    """Remove every verifier-created invocation area except `keep`.

    The contract promises that no scratch state survives between invocations;
    the harness wipes its own runtime tree, so the areas the verifier stages must
    be removed here as well before each further invocation (and again after the
    last one), so the candidate can never read an earlier invocation's scratch.
    """
    for area in verifier_invocation_areas():
        if keep is not None and area == keep:
            continue
        remove_tree(area)


def record_invocation_time(name, seconds):
    """Record a verifier-run invocation's wall time for the aggregate budget check."""
    directory = Path("/root/status")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"verifier_{name}.json").write_text(
        json.dumps({"elapsed_seconds": float(seconds)}) + "\n", encoding="utf-8", newline="\n")


def generated_cases(solver_digest=None):
    """The generated cases: a deterministic function of the submitted solver's bytes.

    The export stages the candidate as `/root/candidate.py` under root and records
    its SHA-256 in `/root/export.json`, so each submission always faces the same
    two instances and the verifier's verdict is reproducible; different
    submissions face different instances, so no precomputed table or input
    fingerprint can answer them.  Both sizes are fixed (8 and 12 hubs: a mid-size
    instance and a larger one) so the verifier cost and the recovery depth
    exercised are bounded and known.  Every case certifies itself by construction
    inside `build_generated_instance` before the candidate runs.
    """
    if solver_digest is None:
        try:
            recorded = read_json(Path("/root/export.json")).get("solver_sha256", "")
            solver_digest = bytes.fromhex(recorded) if len(recorded) == 64 else None
        except Exception:
            solver_digest = None
    if not solver_digest:
        solver_digest = hashlib.sha256(b"minimum-dominating-set").digest()
    cases = []
    for index, hubs in enumerate(GENERATED_SIZES):
        seed = int.from_bytes(hashlib.sha256(solver_digest + bytes([index])).digest()[:8], "big")
        cases.append((seed, hubs))
    return cases


def build_generated_instance(hubs, seed):
    """Generate a hub-star instance whose minimum dominating set is exactly `hubs`.

    `hubs` hubs each carry a private leaf block of three to six leaves and are
    adjacent only to their own leaves.  The closed neighbourhoods of the hubs
    are pairwise disjoint and cover every vertex, so any dominating set needs
    one distinct vertex per hub while the hubs themselves attain that bound:
    the minimum is exactly `hubs`.  Every hub's closed neighbourhood is strictly
    larger than any leaf's, so a maximum-gain greedy construction takes hubs
    until every vertex is covered and also reaches `hubs` - the generated cases
    satisfy the contract's greedy-attainability statement for every seed.  Both
    sides of this construction certificate are asserted in this function before
    the instance is returned.
    """
    rng = random.Random(seed)
    blocks = []
    total = 0
    for _ in range(hubs):
        size = 1 + rng.randint(3, 6)
        blocks.append(list(range(total, total + size)))
        total += size
    n = total
    edges = set()
    for block in blocks:
        hub = block[0]
        for leaf in block[1:]:
            edges.add((hub, leaf) if hub < leaf else (leaf, hub))
    labels = list(range(n))
    rng.shuffle(labels)
    graph = sorted(tuple(sorted((labels[u], labels[v]))) for u, v in edges)
    document = {"schema_version": SCHEMA, "instances": [{"n": n, "edges": [list(edge) for edge in graph]}]}

    # construction certificate: the hub closed neighbourhoods are pairwise
    # disjoint and cover every vertex, so exactly `hubs` vertices are needed and
    # sufficient
    hub_labels = {labels[block[0]] for block in blocks}
    adjacency = [set() for _ in range(n)]
    for u, v in graph:
        adjacency[u].add(v)
        adjacency[v].add(u)
    covered = set()
    for hub in hub_labels:
        closed = {hub} | adjacency[hub]
        assert not (covered & closed), "generated hub neighbourhoods are not disjoint"
        covered |= closed
    assert covered == set(range(n)), "generated hubs do not cover every vertex"
    assert len(hub_labels) == hubs
    return document, hubs


def test_candidate_derives_answers_from_input():
    """The candidate derives answers for instance sets generated at verification time.

    The instruction requires every answer to be derived from the input rather
    than replayed.  The packaged sets are fixed, so a solver carrying precomputed
    answers for them could pass the packaged cases; this test therefore builds two
    further small instances as a deterministic function of the submitted solver's
    own bytes (the same submission always faces the same two instances, so the
    verdict is reproducible; different submissions face different instances, so
    no precomputed table or input fingerprint can answer them), at two fixed hub
    counts (8 and 12: a mid-size case and a larger case, bounding the verifier
    cost and the recovery depth exercised).  Every instance is certified by
    construction - the hubs dominate and their closed neighbourhoods are
    pairwise disjoint, both asserted before the candidate runs - so the verdict
    is exact for every instance.  The candidate is invoked through the harness's
    own sandboxed path (unprivileged user, network block, resource caps, fresh
    scratch, and the prior invocations' runtime tree wiped first) and must write
    a valid minimal answer for each; the verifier-staged areas are swept between
    invocations so no earlier scratch survives.
    """
    harness = load_harness()
    assert harness.solver_network_blocked() is True, (
        "the network-prevention probe did not confirm the seccomp block for a generated invocation")
    for index, (seed, hubs) in enumerate(generated_cases()):
        document, optimum_size = build_generated_instance(hubs, seed)
        validate_input(document)
        area = Path(f"/root/generated-{index}")
        harness.clear_runtime_tree()
        clear_other_invocation_areas(keep=area)
        remove_tree(area)
        area.mkdir(parents=True)
        os.chmod(area, 0o755)
        staged = area / "generated_instances.json"
        staged.write_bytes((json.dumps(document, sort_keys=True) + "\n").encode("utf-8"))
        os.chmod(staged, 0o444)
        staged_output = area / "output"
        runtime = area / "runtime"
        scratch = area / "scratch"
        logs = area / "logs"
        for directory in (staged_output, runtime, scratch):
            directory.mkdir()
            os.chown(directory, 65534, 65534, follow_symlinks=False)
            os.chmod(directory, 0o700, follow_symlinks=False)
        logs.mkdir()
        execution = harness.execute_solver(
            staged, staged_output, runtime, scratch, logs / "stdout.log", logs / "stderr.log")
        record_invocation_time(f"generated-{index}", execution["elapsed_seconds"])
        stderr_tail = (logs / "stderr.log").read_text(encoding="utf-8", errors="replace")[-4000:]
        assert execution["timed_out"] is False, (
            "the generated-instance invocation exceeded the 3600-second solver limit")
        assert execution["cleanup_ok"] is True, (
            "the generated-instance invocation left an unprivileged process running")
        assert execution["returncode"] == 0, (
            f"the candidate returned {execution['returncode']} on a generated instance set; "
            f"required status is 0. Solver stderr tail:\n{stderr_tail}")
        violations = harness.scan_output_directory(staged_output)
        assert violations == [], (
            f"the generated output directory contains rejected entries: {violations}")
        candidate = read_json(staged_output / "solution.json", expected_directory=staged_output)
        validate_solution(document, candidate, {0: optimum_size})
    # Leave no scratch behind once every verifier invocation has finished.
    clear_other_invocation_areas()


def test_aggregate_wall_clock_budget_is_enforced():
    """The combined wall-clock limit covers every invocation the verifier runs.

    The contract requires the solver to complete every invocation - the
    public call, the unseen call and the generated-input calls - inside a
    combined 14,400-second wall-clock limit.  This test sums the measured wall
    time of all four invocations and fails the run when the total exceeds that
    limit, so individually admissible slow runs cannot pass together.
    """
    total = 0.0
    for split in SPLITS:
        status = read_json(Path("/root/status") / f"{split}.json")
        total += float(status["elapsed_seconds"])
    for name in [f"generated-{index}" for index in range(len(GENERATED_SIZES))]:
        path = Path("/root/status") / f"verifier_{name}.json"
        if not path.is_file():
            # The invocation failed its own assertion before recording its time;
            # the run is already failing and there is nothing to add here.
            continue
        total += float(read_json(path)["elapsed_seconds"])
    assert total <= 14400.0, (
        f"the verifier-run invocations took {total:.1f} s in total; the combined wall-clock limit is 14,400 s")

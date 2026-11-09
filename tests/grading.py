"""Exact scientific validation for the minimum dominating set task.

The verifier re-derives every requirement from the input graph: a candidate
answer must list distinct in-range vertices, its closed neighbourhoods must
cover every vertex, and its size must not exceed the independently certified
minimum size for that instance.  Read paths are strict: JSON with duplicate
keys or non-finite numbers, quoted CSV fields and non-canonical integers are
rejected with measured values in the message.  Candidate-controlled files are
read symlink-proof: the path must stay inside the expected output directory and
the bytes are read from a descriptor that was opened without following links and
re-checked with fstat.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import stat
from pathlib import Path

SCHEMA = "minimum-dominating-set-v1"
OUTPUT_KEYS = {"dominating_set"}
INPUT_KEYS = {"n", "edges"}
MAX_N = 40000
MAX_EDGES = 600000


class ContractError(Exception):
    pass


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ContractError(f"non-finite JSON number: {value}")


# Symlink-proof reading of candidate-controlled paths.  O_NOFOLLOW is not
# available on every platform; the lstat check plus the fstat re-check on the
# opened descriptor still reject links, and the realpath containment check
# rejects links in any path component that would escape the expected output
# directory.
O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
MAX_READ_BYTES = 64_000_000


def read_regular_bytes(path: Path, expected_directory: Path | None = None, label: str = "file") -> bytes:
    """Read a regular file without ever following a link.

    When `expected_directory` is given (a candidate-controlled path), the
    realpath must stay inside that directory, the final component must not be a
    symlink (rejected with lstat and again by opening with O_NOFOLLOW), and the
    bytes are read from the opened descriptor so the object re-checked with
    fstat is exactly the object read.
    """
    if expected_directory is not None:
        directory = os.path.realpath(expected_directory)
        resolved = os.path.realpath(path)
        if resolved != directory and not resolved.startswith(directory + os.sep):
            raise ContractError(
                f"{label} resolves to {resolved}, outside the expected output directory {directory}")
    try:
        metadata = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        raise ContractError(f"{label} is missing: {path}")
    if stat.S_ISLNK(metadata.st_mode):
        raise ContractError(f"{label} is a symbolic link; links are not accepted: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"{label} is not a regular file: {path}")
    if metadata.st_size > MAX_READ_BYTES:
        raise ContractError(f"{label} has size {metadata.st_size} bytes; required limit is {MAX_READ_BYTES} bytes")
    try:
        descriptor = os.open(path, os.O_RDONLY | O_NOFOLLOW)
    except OSError as exc:
        raise ContractError(f"{label} cannot be opened safely: {path}: {type(exc).__name__}: {exc}")
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ContractError(f"{label} is not a regular file: {path}")
        if opened.st_size > MAX_READ_BYTES:
            raise ContractError(f"{label} has size {opened.st_size} bytes; required limit is {MAX_READ_BYTES} bytes")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def read_json(path: Path, expected_directory: Path | None = None) -> dict:
    data = read_regular_bytes(path, expected_directory, label="JSON file")
    try:
        text = data.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"invalid strict JSON in {path}: {type(exc).__name__}: {exc}")
    if not isinstance(value, dict):
        raise ContractError("top-level JSON value must be an object")
    return value


def parse_optimum(path: Path, expected_directory: Path | None = None) -> dict[int, int]:
    data = read_regular_bytes(path, expected_directory, label="independent optimum table")
    text = data.decode("utf-8")
    _lines = text.split("\n")
    required_header = "instance,minimum_size"
    if not _lines or _lines[0] != required_header:
        measured_header = _lines[0] if _lines else None
        raise ContractError(f"optimum.csv raw header is {measured_header!r}; required header is {required_header}")
    rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    result = {}
    for line_number, (raw, row) in enumerate(zip(_lines, rows), 1):
        if raw.split(",") != row:
            raise ContractError(f"optimum.csv line {line_number} contains CSV-quoted or normalized fields")
        if line_number == 1:
            continue
        if len(row) != 2 or not re.fullmatch(r"(?:0|[1-9][0-9]*)", row[0]) or not re.fullmatch(r"(?:0|[1-9][0-9]*)", row[1]):
            raise ContractError(f"optimum.csv line {line_number} must contain an instance index and a canonical minimum size")
        index = int(row[0])
        if index in result:
            raise ContractError(f"optimum.csv repeats instance {index}")
        result[index] = int(row[1])
    if len(_lines) != len(rows) + 1 or _lines[-1] != "":
        raise ContractError(f"optimum.csv has {len(rows)} parsed rows and final spelling {_lines[-1]!r}; required layout is one terminal LF")
    if not result:
        raise ContractError("optimum.csv contains no data rows")
    return result


def validate_input(document: dict) -> None:
    if set(document) != {"schema_version", "instances"} or document.get("schema_version") != SCHEMA:
        raise ContractError(f"input must have exactly schema_version/instances and schema {SCHEMA}")
    instances = document["instances"]
    if not isinstance(instances, list) or not instances:
        raise ContractError("input must contain a non-empty instance list")
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict) or set(instance) != INPUT_KEYS:
            raise ContractError(f"instance {index} keys must be exactly {sorted(INPUT_KEYS)}")
        n = instance["n"]
        if not isinstance(n, int) or isinstance(n, bool) or not 2 <= n <= MAX_N:
            raise ContractError(f"instance {index}: n={n!r}; required range is 2 through {MAX_N}")
        edges = instance["edges"]
        if not isinstance(edges, list):
            raise ContractError(f"instance {index}: edges must be an array")
        if len(edges) > MAX_EDGES:
            raise ContractError(f"instance {index}: {len(edges)} edges; required limit is {MAX_EDGES}")
        seen = set()
        for position, edge in enumerate(edges):
            if not isinstance(edge, list) or len(edge) != 2:
                raise ContractError(f"instance {index} edge {position}: each edge must be a two-element array")
            u, v = edge
            for value in (u, v):
                if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < n:
                    raise ContractError(f"instance {index} edge {position}: endpoint {value!r} is outside 0..{n-1}")
            if u == v:
                raise ContractError(f"instance {index} edge {position}: self loop on vertex {u}")
            key = (u, v) if u < v else (v, u)
            if key in seen:
                raise ContractError(f"instance {index} edge {position}: duplicate edge {key}")
            seen.add(key)


def _validate_answer(index: int, instance: dict, answer, minimum: int) -> None:
    """Validate one answer; raises ContractError naming the failed requirement."""
    if not isinstance(answer, dict) or set(answer) != OUTPUT_KEYS:
        raise ContractError(f"answer {index} must have exactly {sorted(OUTPUT_KEYS)}")
    n = instance["n"]
    chosen = answer["dominating_set"]
    if not isinstance(chosen, list):
        raise ContractError(f"answer {index}: dominating_set must be an array")
    if not chosen:
        raise ContractError(f"answer {index}: dominating_set is empty")
    seen = set()
    for position, value in enumerate(chosen):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ContractError(f"answer {index} entry {position}: vertex {value!r} is not an integer")
        if not 0 <= value < n:
            raise ContractError(f"answer {index} entry {position}: vertex {value} is outside 0..{n-1}")
        if value in seen:
            raise ContractError(f"answer {index} entry {position}: repeated vertex {value}")
        seen.add(value)
    chosen_set = set(chosen)
    dominated = set(chosen_set)
    for u, v in instance["edges"]:
        if u in chosen_set:
            dominated.add(v)
        if v in chosen_set:
            dominated.add(u)
    if len(dominated) != n:
        for vertex in range(n):
            if vertex not in dominated:
                raise ContractError(
                    f"answer {index}: vertex {vertex} is neither chosen nor adjacent to a chosen vertex")
        raise ContractError(f"answer {index}: {n - len(dominated)} vertices are not dominated")
    if len(chosen) > minimum:
        raise ContractError(
            f"answer {index}: measured dominating size {len(chosen)}; independently required minimum is {minimum}")


def validate_solution(document: dict, solution: dict, optimum: dict[int, int]) -> None:
    if set(solution) != {"instances"}:
        raise ContractError("solution must have exactly the instances key")
    instances = document["instances"]
    answers = solution["instances"]
    if not isinstance(answers, list) or len(answers) != len(instances):
        raise ContractError(
            f"solution contains {len(answers) if isinstance(answers, list) else 'non-array'} entries; required count is {len(instances)}")
    if set(optimum) != set(range(len(instances))):
        raise ContractError("independent optimum roster does not match the input instance roster")
    # Validate every instance and collect all failures before raising, so a
    # multi-instance log names every instance that failed instead of only the
    # first one.  Instances that pass are simply absent from the failure list.
    failures = []
    for index, (instance, answer) in enumerate(zip(instances, answers)):
        try:
            _validate_answer(index, instance, answer, optimum[index])
        except ContractError as exc:
            failures.append(str(exc))
    if failures:
        raise ContractError(
            f"{len(failures)} of {len(instances)} instances failed:\n" + "\n".join(failures))


def validate_directory(data_path: Path, output_directory: Path) -> None:
    document = read_json(data_path)
    validate_input(document)
    solution = read_json(output_directory / "solution.json", expected_directory=output_directory)
    optimum = parse_optimum(output_directory / "optimum.csv", expected_directory=output_directory)
    validate_solution(document, solution, optimum)

#!/usr/bin/env python3
"""Reference solver for the minimum dominating set task.

Development instances are handled by maximum-gain greedy construction followed
by redundancy elimination.  Instances whose every vertex has the same degree d
with n = 2**d are treated as scrambled hypercubes: the coordinate system is
rebuilt level by level (a vertex in level L+1 is assigned the unique direction
that keeps it adjacent to the required L vertices of level L), the
sphere-covering lower bound n/(d+1) is computed, and the binary Hamming code
constructed from the canonical parity-check columns 1..d supplies a dominating
set meeting that bound.
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path


def adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def greedy_dominating(n, adj):
    covered = [False] * n
    remaining = n
    chosen = []
    while remaining:
        best = -1
        best_gain = 0
        for v in range(n):
            if v in chosen:
                continue
            gain = 0 if covered[v] else 1
            for u in adj[v]:
                if not covered[u]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best = v
        if best < 0:
            break
        chosen.append(best)
        if not covered[best]:
            covered[best] = True
            remaining -= 1
        for u in adj[best]:
            if not covered[u]:
                covered[u] = True
                remaining -= 1
    return chosen


def eliminate_redundant(n, adj, chosen):
    counts = [0] * n
    for v in chosen:
        counts[v] += 1
        for u in adj[v]:
            counts[u] += 1
    chosen = list(chosen)
    for index in sorted(range(len(chosen)), key=lambda i: -len(adj[chosen[i]])):
        v = chosen[index]
        removable = True
        for w in [v] + adj[v]:
            if counts[w] <= 1:
                removable = False
                break
        if removable:
            counts[v] -= 1
            for u in adj[v]:
                counts[u] -= 1
            chosen[index] = None
    return [v for v in chosen if v is not None]


def local_improve(n, adj, chosen):
    """Swap one chosen vertex for two others while the set stays smaller."""
    chosen = set(eliminate_redundant(n, adj, chosen))
    improved = True
    while improved:
        improved = False
        counts = [0] * n
        for v in chosen:
            counts[v] += 1
            for u in adj[v]:
                counts[u] += 1
        uncovered = [v for v in range(n) if counts[v] == 0]
        if not uncovered:
            break
        candidates = set()
        for v in uncovered:
            candidates.update(adj[v])
            candidates.add(v)
        for v in sorted(chosen, key=lambda x: len(adj[x])):
            trial = (chosen - {v}) | {w for w in list(candidates)[:2]}
            if len(trial) >= len(chosen):
                continue
            counts_trial = [0] * n
            for u in trial:
                counts_trial[u] += 1
                for w in adj[u]:
                    counts_trial[w] += 1
            if min(counts_trial) >= 1:
                chosen = trial
                improved = True
                break
    return sorted(chosen)


def bfs_levels(n, adj):
    level = [-1] * n
    level[0] = 0
    queue = deque([0])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for u in adj[v]:
            if level[u] < 0:
                level[u] = level[v] + 1
                queue.append(u)
    return level, order


def hypercube_coordinates(n, adj):
    """Rebuild hypercube coordinates level by level; returns (degree, coord)."""
    degree = len(adj[0])
    if degree < 2 or n != 1 << degree:
        return None
    if any(len(adj[v]) != degree for v in range(n)):
        return None
    level, order = bfs_levels(n, adj)
    if min(level) < 0 or max(level) != degree:
        return None
    neighbour_sets = [set(adj[v]) for v in range(n)]
    coord = [-1] * n
    coord[0] = 0
    index_of = {0: 0}
    directions = sorted(adj[0])
    for index, u in enumerate(directions):
        coord[u] = 1 << index
        index_of[1 << index] = u
    by_level = {}
    for v in order:
        by_level.setdefault(level[v], []).append(v)
    for depth in range(1, degree):
        for y in by_level.get(depth + 1, []):
            found = -1
            for x in adj[y]:
                if coord[x] < 0 or level[x] != depth:
                    continue
                present = [1 << j for j in range(degree) if (coord[x] >> j) & 1]
                for k in range(degree):
                    if (coord[x] >> k) & 1:
                        continue
                    candidate = coord[x] ^ (1 << k)
                    valid = True
                    for jbit in present:
                        partner = coord[x] ^ jbit ^ (1 << k)
                        vertex = index_of.get(partner)
                        if vertex is None or vertex not in neighbour_sets[y]:
                            valid = False
                            break
                    if valid:
                        found = candidate
                        break
                if found >= 0:
                    break
            if found < 0 or found in index_of:
                return None
            coord[y] = found
            index_of[found] = y
    if any(value < 0 for value in coord) or len(set(coord)) != n:
        return None
    for v in range(n):
        for u in adj[v]:
            delta = coord[u] ^ coord[v]
            if delta == 0 or (delta & (delta - 1)):
                return None
    return degree, coord


def hamming_code(dim):
    """Perfect code: kernel of the parity-check matrix with columns 1..dim."""
    height = dim.bit_length()
    if (1 << height) - 1 != dim:
        return None
    code = []
    for x in range(1 << dim):
        good = True
        for row in range(height):
            parity = 0
            for value in range(1, dim + 1):
                if (x >> (value - 1)) & 1 and (value >> row) & 1:
                    parity ^= 1
            if parity:
                good = False
                break
        if good:
            code.append(x)
    return code


def exact_dominating(n, adj):
    """Complete branch and bound used for small instances."""
    best = greedy_dominating(n, adj)
    best = eliminate_redundant(n, adj, best)
    best_size = len(best)
    closed = [[v] + adj[v] for v in range(n)]
    order = sorted(range(n), key=lambda v: len(closed[v]))
    full = (1 << n) - 1
    current = []

    def search(dominated, depth):
        nonlocal best, best_size
        if dominated == full:
            if depth < best_size:
                best = list(current)
                best_size = depth
            return
        if depth + 1 >= best_size:
            return
        target = -1
        for v in order:
            if not ((dominated >> v) & 1):
                target = v
                break
        for u in closed[target]:
            added = 0
            for w in closed[u]:
                if not ((dominated >> w) & 1):
                    added |= 1 << w
            current.append(u)
            search(dominated | added, depth + 1)
            current.pop()

    search(0, 0)
    return best


def disjoint_closed_cover(n, adj):
    """A closed-size class whose closed neighbourhoods partition the vertex set.

    Closed neighbourhoods that are pairwise disjoint form a 2-packing, so every
    dominating set contains at least one vertex per element; when the same
    family covers the whole graph, the two bounds meet and the family is a
    minimum dominating set.  Every distinct closed size is tried, so pure
    hub/leaf graphs are recognized even when the hubs are not the
    minimum-closed-degree vertices.
    """
    sizes = [len(adj[v]) + 1 for v in range(n)]
    by_size = {}
    for v, size in enumerate(sizes):
        by_size.setdefault(size, []).append(v)
    for size in sorted(by_size):
        candidates = by_size[size]
        covered = bytearray(n)
        ok = True
        for v in candidates:
            if covered[v]:
                ok = False
                break
            covered[v] = 1
            for u in adj[v]:
                if covered[u]:
                    ok = False
                    break
                covered[u] = 1
            if not ok:
                break
        if ok and all(covered):
            return candidates
    return None


def solve_instance(n, edges):
    adj = adjacency(n, edges)
    if n > 4096:
        cube = hypercube_coordinates(n, adj)
        if cube is not None:
            dim, coord = cube
            code = hamming_code(dim)
            if code is not None:
                index_of = {value: v for v, value in enumerate(coord)}
                chosen = [index_of[value] for value in code]
                counts = [0] * n
                for v in chosen:
                    counts[v] += 1
                    for u in adj[v]:
                        counts[u] += 1
                if min(counts) >= 1 and len(chosen) == -(-n // (dim + 1)):
                    return chosen
    selected = disjoint_closed_cover(n, adj)
    if selected is not None:
        return selected
    if n <= 4096:
        return exact_dominating(n, adj)
    chosen = greedy_dominating(n, adj)
    chosen = local_improve(n, adj, chosen)
    return chosen


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: dominating_reference.py INPUT.json OUTPUT_DIRECTORY")
    document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    solutions = []
    for instance in document["instances"]:
        chosen = solve_instance(instance["n"], instance["edges"])
        solutions.append({"dominating_set": sorted(chosen)})
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    (output / "solution.json").write_text(
        json.dumps({"instances": solutions}, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("sizes", [len(entry["dominating_set"]) for entry in solutions], flush=True)
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    (output / "solution.json").write_text(
        json.dumps({"instances": solutions}, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("sizes", [len(entry["dominating_set"]) for entry in solutions], flush=True)


if __name__ == "__main__":
    main()

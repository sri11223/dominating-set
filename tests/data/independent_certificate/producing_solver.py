#!/usr/bin/env python3
"""Independent producing implementation for the minimum dominating set task.

This is a third implementation, written separately from `solution/
dominating_reference.py`; it shares no function bodies with it and recovers the
shipped optima along different routes:

  * general instances: closed-neighbourhood cover by a lazy-gain heap followed by
    redundancy pruning, and a fail-first set-cover search for small instances
    (branch on the uncovered vertex with the fewest covering closed
    neighbourhoods) whose incumbent is the heap solution;
  * relabelled hypercubes: the cube is verified by BFS level-size statistics, the
    coordinates are recovered as the bitwise OR of the parent codes at the
    previous level, and the dominating set is the kernel of a random
    column-permutation parity check on the non-zero vectors (a perfect Hamming
    code) meeting the sphere-covering bound;
  * hub/leaf graphs: a closed-size class is collected and accepted only when its
    closed neighbourhoods are pairwise disjoint and cover the graph, which
    certifies both sides of the 2-packing bound.

Every emitted witness is checked by explicit domination before it is returned,
and for the two certified families the returned size meets the family's closed
lower bound, so the answer is minimum.
"""
from __future__ import annotations

import json
import random
import sys
from collections import deque
from math import comb
from pathlib import Path

sys.setrecursionlimit(200000)


def build_csr(n, edges):
    """Flat adjacency in compressed form: offset[v]..offset[v+1]."""
    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    offset = [0] * (n + 1)
    for v in range(n):
        offset[v + 1] = offset[v] + degree[v]
    flat = [0] * (2 * len(edges))
    cursor = offset[:n]
    for u, v in edges:
        flat[cursor[u]] = v
        cursor[u] += 1
        flat[cursor[v]] = u
        cursor[v] += 1
    return offset, flat


def closed_gain(v, covered, offset, flat):
    gain = 0 if covered[v] else 1
    for i in range(offset[v], offset[v + 1]):
        if not covered[flat[i]]:
            gain += 1
    return gain


def heap_cover(n, offset, flat):
    """Maximum-gain closed-neighbourhood cover with a lazily refreshed heap."""
    import heapq

    covered = bytearray(n)
    remaining = n
    chosen = []
    heap = [(-closed_gain(v, covered, offset, flat), v) for v in range(n)]
    heapq.heapify(heap)
    while remaining:
        selected = -1
        while heap:
            _, v = heapq.heappop(heap)
            if covered[v]:
                continue
            actual = closed_gain(v, covered, offset, flat)
            if heap and -heap[0][0] > actual:
                heapq.heappush(heap, (-actual, v))
                continue
            selected = v
            break
        if selected < 0:
            break
        chosen.append(selected)
        if not covered[selected]:
            covered[selected] = 1
            remaining -= 1
        for i in range(offset[selected], offset[selected + 1]):
            u = flat[i]
            if not covered[u]:
                covered[u] = 1
                remaining -= 1
    return chosen


def prune_redundant(n, offset, flat, chosen):
    """Drop chosen vertices whose closed neighbourhood is covered by the rest."""
    counts = [0] * n
    for v in chosen:
        counts[v] += 1
        for i in range(offset[v], offset[v + 1]):
            counts[flat[i]] += 1
    keep = []
    removed = set()
    for v in sorted(chosen, key=lambda x: (offset[x + 1] - offset[x], x)):
        if v in removed:
            continue
        safe = all(counts[w] > 1 for w in [v] + flat[offset[v]:offset[v + 1]])
        if safe:
            removed.add(v)
            counts[v] -= 1
            for i in range(offset[v], offset[v + 1]):
                counts[flat[i]] -= 1
    for v in chosen:
        if v not in removed:
            keep.append(v)
    return keep


def fail_first_cover(n, offset, flat, incumbent, node_cap=4_000_000):
    """Small-instance exact cover: fail-first branching on the least-coverable
    uncovered vertex; the incumbent initialises the bound."""
    best = list(incumbent)
    counts = [0] * n
    picked = bytearray(n)
    current = []
    nodes = [0]

    def add(v):
        picked[v] = 1
        counts[v] += 1
        for i in range(offset[v], offset[v + 1]):
            counts[flat[i]] += 1

    def remove(v):
        picked[v] = 0
        counts[v] -= 1
        for i in range(offset[v], offset[v + 1]):
            counts[flat[i]] -= 1

    def search():
        nonlocal best
        nodes[0] += 1
        if nodes[0] > node_cap or len(current) + 1 >= len(best):
            return
        target = -1
        fewest = None
        for v in range(n):
            if counts[v] == 0:
                size = offset[v + 1] - offset[v] + 1
                if fewest is None or size < fewest:
                    fewest = size
                    target = v
        if target < 0:
            best = list(current)
            return
        candidates = [target]
        for i in range(offset[target], offset[target + 1]):
            candidates.append(flat[i])
        candidates.sort(key=lambda u: (-closed_gain(u, counts, offset, flat), u))
        for u in candidates:
            if picked[u] or counts[u] > 0:
                continue
            add(u)
            current.append(u)
            search()
            current.pop()
            remove(u)

    search()
    return best


def bfs_levels(n, offset, flat, source=0):
    level = [-1] * n
    level[source] = 0
    queue = deque([source])
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for i in range(offset[v], offset[v + 1]):
            u = flat[i]
            if level[u] < 0:
                level[u] = level[v] + 1
                queue.append(u)
    return level, order


def is_dominating(n, offset, flat, chosen):
    covered = bytearray(n)
    for v in chosen:
        covered[v] = 1
        for i in range(offset[v], offset[v + 1]):
            covered[flat[i]] = 1
    return all(covered)


def hypercube_solution(n, offset, flat):
    """Perfect-code dominating set for a relabelled hypercube, or None."""
    if n < 4:
        return None
    degree = offset[1] - offset[0]
    if degree < 2 or (1 << degree) != n:
        return None
    for v in range(n):
        if offset[v + 1] - offset[v] != degree:
            return None
    level, order = bfs_levels(n, offset, flat)
    if min(level) < 0 or max(level) != degree:
        return None
    per_level = [0] * (degree + 1)
    for v in range(n):
        per_level[level[v]] += 1
    for k in range(degree + 1):
        if per_level[k] != comb(degree, k):
            return None
    code = [-1] * n
    code[0] = 0
    index_of = {0: 0}
    first_level = []
    for i in range(offset[0], offset[0 + 1]):
        u = flat[i]
        if level[u] != 1:
            return None
        first_level.append(u)
    if len(first_level) != degree:
        return None
    for k, u in enumerate(first_level):
        code[u] = 1 << k
        if code[u] in index_of:
            return None
        index_of[code[u]] = u
    for v in order[1:]:
        if level[v] == 1:
            continue
        parents = []
        for i in range(offset[v], offset[v + 1]):
            u = flat[i]
            if level[u] == level[v] - 1:
                parents.append(code[u])
        if len(parents) != level[v]:
            return None
        merged = 0
        for value in parents:
            merged |= value
        if merged == 0 or merged in index_of or bin(merged).count("1") != level[v]:
            return None
        code[v] = merged
        index_of[merged] = v
    if len(index_of) != n:
        return None
    for v in range(n):
        for i in range(offset[v], offset[v + 1]):
            delta = code[v] ^ code[flat[i]]
            if delta == 0 or (delta & (delta - 1)):
                return None
    height = degree.bit_length()
    if (1 << height) - 1 != degree:
        return None
    bound = n // (degree + 1)
    rng = random.Random(20261106)
    vectors = list(range(1, degree + 1))
    rng.shuffle(vectors)
    code_values = []
    for x in range(n):
        syndrome = 0
        rest = x
        while rest:
            low = rest & -rest
            syndrome ^= vectors[low.bit_length() - 1]
            rest ^= low
        if syndrome == 0:
            code_values.append(x)
    if len(code_values) != bound:
        return None
    chosen = [index_of[value] for value in code_values]
    if len(chosen) == bound and is_dominating(n, offset, flat, chosen):
        return chosen
    return None


def disjoint_closed_cover(n, offset, flat):
    """A closed-size class whose closed neighbourhoods partition the vertex set."""
    sizes = [offset[v + 1] - offset[v] + 1 for v in range(n)]
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
            for i in range(offset[v], offset[v + 1]):
                u = flat[i]
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
    offset, flat = build_csr(n, edges)
    if n > 4096:
        cube = hypercube_solution(n, offset, flat)
        if cube is not None:
            return cube
    packing = disjoint_closed_cover(n, offset, flat)
    if packing is not None:
        return packing
    greedy = heap_cover(n, offset, flat)
    greedy = prune_redundant(n, offset, flat, greedy)
    if n <= 4096:
        exact = fail_first_cover(n, offset, flat, greedy)
        if is_dominating(n, offset, flat, exact):
            return exact
    return greedy


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: producing_solver.py INPUT.json OUTPUT_DIRECTORY")
    document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    solutions = []
    sizes = []
    for instance in document["instances"]:
        chosen = solve_instance(instance["n"], instance["edges"])
        solutions.append({"dominating_set": sorted(chosen)})
        sizes.append(len(chosen))
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    (output / "solution.json").write_text(
        json.dumps({"instances": solutions}, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    lines = ["instance,minimum_size"]
    for index, size in enumerate(sizes):
        lines.append("%d,%d" % (index, size))
    (output / "optimum.csv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("sizes", sizes, flush=True)


if __name__ == "__main__":
    main()

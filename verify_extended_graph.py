"""
verify_extended_graph.py
Checks the extended-graph construction for odd n = 5, 7, 9, 11.
Run: python verify_extended_graph.py
"""
import numpy as np
from numpy.linalg import svd, norm, eigvalsh


def symmetric_cyclic_vectors(n):
    """Symmetric cyclic configuration for C_n."""
    m = (n - 1) // 2
    cos_t2 = np.cos(np.pi / n) / (1 + np.cos(np.pi / n))
    cos_t = np.sqrt(cos_t2)
    sin_t = np.sqrt(1 - cos_t2)
    return [np.array([cos_t,
                      sin_t * np.cos(2 * np.pi * i * m / n),
                      sin_t * np.sin(2 * np.pi * i * m / n)],
                     dtype=complex)
            for i in range(n)]


def orth_complement_1d(a, b):
    """Unique direction (up to phase) orthogonal to both a and b."""
    M = np.vstack([np.conj(a), np.conj(b)])
    _, _, Vh = svd(M)
    v = Vh[-1].conj()
    return v / norm(v)


def build_extended(n):
    """Build base + extended vectors."""
    base = symmetric_cyclic_vectors(n)
    k = (n + 1) // 2
    ext = [orth_complement_1d(base[0], base[n - 1])]
    for j in range(1, k):
        # Extended vertex n + j (0-indexed: n + j - 1 in base indexing)
        # is orthogonal to base vertices at indices 2j-1 and 2j.
        ext.append(orth_complement_1d(base[2 * j - 1], base[2 * j]))
    return base + ext


def check_orthogonality(vectors, n):
    """Return set of all pairs with |<v_i|v_j>| < 1e-10."""
    k = (n + 1) // 2
    N = n + k
    orth_pairs = set()
    for i in range(N):
        for j in range(i + 1, N):
            ov = abs(np.vdot(vectors[i], vectors[j]))
            if ov < 1e-10:
                orth_pairs.add((i, j))
    return orth_pairs


def target_edges(n):
    """Edges required by the construction (0-indexed)."""
    k = (n + 1) // 2
    edges = set()

    # Base cycle edges: (i, i+1) and (n-1, 0)
    for i in range(n):
        edges.add(tuple(sorted((i, (i + 1) % n))))

    # Extended vertex n (0-indexed) closes triangle {0, n-1, n}
    ext0 = n  # first extended vertex
    edges.add(tuple(sorted((0, ext0))))
    edges.add(tuple(sorted((n - 1, ext0))))

    # Extended vertices n+1, ..., n+k-1 close triangles {2j-1, 2j, n+j}
    for j in range(1, k):
        ext_v = n + j
        edges.add(tuple(sorted((2 * j - 1, ext_v))))
        edges.add(tuple(sorted((2 * j, ext_v))))

    return edges


def main():
    for n in [5, 7, 9, 11]:
        print(f"\n=== n = {n} ===")
        vectors = build_extended(n)
        k = (n + 1) // 2
        print(f"  Total vectors: {n + k}")

        orth = check_orthogonality(vectors, n)
        target = target_edges(n)

        missing = target - orth
        extra = orth - target

        print(f"  Target edges: {len(target)}")
        print(f"  Orthogonal pairs found: {len(orth)}")
        print(f"  Missing (required but not orthogonal): {len(missing)}")
        print(f"  Extra (orthogonal but not required): {len(extra)}")

        if missing:
            print(f"  *** FAILED: Missing edges (first 5): {sorted(missing)[:5]}")
        elif extra:
            print(f"  *** WARNING: Extra orthogonalities present ***")
            print(f"  Extra pairs (first 5): {sorted(extra)[:5]}")
        else:
            print(f"  *** PASSED: Construction exactly realizes the graph ***")

        # Check witness bound for the base n vectors
        P = [np.outer(v, np.conj(v)) for v in vectors[:n]]
        W = sum(P)
        beta = float(np.max(eigvalsh(W)))
        beta_theory = n * np.cos(np.pi / n) / (1 + np.cos(np.pi / n))
        print(f"  beta numerical  = {beta:.6f}")
        print(f"  beta theoretical = {beta_theory:.6f}")
        print(f"  Error = {abs(beta - beta_theory):.2e}")


if __name__ == "__main__":
    main()
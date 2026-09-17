"""
COMPARISON OF NOISE MODELS FOR KCBS PROTOCOL
Run: python noise_models_final.py

Verifies the ideal case first, then computes noise thresholds for
depolarizing, amplitude damping, dephasing, and bit-flip noise models.
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# KCBS VECTORS + EXTENDED VERTICES
# ============================================================

def kcbs_vectors():
    """KCBS 5-cycle vectors in C^3 with cos^2(theta) = 1/sqrt(5)."""
    cos_t = np.sqrt(1.0 / np.sqrt(5))
    sin_t = np.sqrt(1.0 - 1.0 / np.sqrt(5))
    vectors = []
    for i in range(5):
        phi = 4 * np.pi * i / 5
        v = np.array([cos_t,
                      sin_t * np.cos(phi),
                      sin_t * np.sin(phi)], dtype=complex)
        vectors.append(v)
    return vectors

def orthogonal_vector(a, b):
    """Find unit vector orthogonal to both a and b in C^3."""
    M = np.vstack([np.conj(a), np.conj(b)])
    U, S, Vh = np.linalg.svd(M)
    v = Vh[-1].conj()
    return v / np.linalg.norm(v)

def build_all_projectors():
    """
    Build 5 original KCBS vectors + 3 extended vectors.
    
    Extended structure (from paper):
    - v6 orthogonal to v1 and v5 (creates triangle 1-5-6)
    - v7 orthogonal to v2 and v3 (creates triangle 2-3-7)
    - v8 orthogonal to v4 and v5 (creates triangle 4-5-8)
    """
    vectors = kcbs_vectors()
    v6 = orthogonal_vector(vectors[0], vectors[4])  # ⊥ v1, v5
    v7 = orthogonal_vector(vectors[1], vectors[2])  # ⊥ v2, v3
    v8 = orthogonal_vector(vectors[3], vectors[4])  # ⊥ v4, v5
    all_vecs = vectors + [v6, v7, v8]
    all_projs = [np.outer(v, np.conj(v)) for v in all_vecs]
    return all_vecs, all_projs

# ============================================================
# EXTENDED GRAPH ADJACENCY (0-indexed, 8 vertices)
# ============================================================

ADJ = {
    0: [1, 4, 5],
    1: [0, 2, 6],
    2: [1, 3, 6],
    3: [2, 4, 7],
    4: [3, 0, 5, 7],
    5: [0, 4],
    6: [1, 2],
    7: [3, 4],
}

N_TOTAL = 35      # 5 + 3 + 22 + 5
S_C = 0.914286
S_BETA = np.sqrt(5)  # ideal quantum bound

# ============================================================
# NOISE CHANNELS
# ============================================================

def apply_depolarizing(rho, mu):
    d = rho.shape[0]
    return mu * rho + (1 - mu) * np.eye(d) / d

def apply_amplitude_damping(rho, gamma):
    K0 = np.array([[1, 0, 0],
                   [0, np.sqrt(1 - gamma), 0],
                   [0, 0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma), 0],
                   [0, 0, 0],
                   [0, 0, 0]], dtype=complex)
    K2 = np.array([[0, 0, np.sqrt(gamma)],
                   [0, 0, 0],
                   [0, 0, 0]], dtype=complex)
    return K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T + K2 @ rho @ K2.conj().T

def apply_dephasing(rho, mu):
    return mu * rho + (1 - mu) * np.diag(np.diag(rho))

def apply_bit_flip(rho, mu):
    X = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=complex)
    X2 = X @ X
    return mu * rho + (1 - mu) / 2 * (X @ rho @ X.conj().T + X2 @ rho @ X2.conj().T)

# ============================================================
# DIRECT SIMULATION OF S_Q
# ============================================================

def compute_SQ(all_projs, noise_fn, param):
    """
    Compute S_Q = (1/N) * [Term1 + Term2 + Term3]
    
    Term1: sum_{x=0..7} p(z=0 | x, y=x)
    Term2: sum_{x=0..7} sum_{y in N_x} p(z=1 | x, y)
    Term3: sum_{y=0..4} w_y * p(z=0 | x=0, y)
    """
    nk = 8

    # Term 1: diagonal terms
    term1 = 0.0
    for x in range(nk):
        noisy = noise_fn(all_projs[x], param)
        term1 += np.real(np.trace(noisy @ all_projs[x]))

    # Term 2: off-diagonal terms
    term2 = 0.0
    for x in range(nk):
        noisy = noise_fn(all_projs[x], param)
        for y in ADJ[x]:
            p0 = np.real(np.trace(noisy @ all_projs[y]))
            term2 += 1 - p0

    # Term 3: contextuality witness term
    W = sum(all_projs[:5])
    eigvals, eigvecs = np.linalg.eigh(W)
    rho_0 = np.outer(eigvecs[:, -1], np.conj(eigvecs[:, -1]))
    noisy_0 = noise_fn(rho_0, param)
    term3 = 0.0
    for y in range(5):
        term3 += np.real(np.trace(noisy_0 @ all_projs[y]))

    return (term1 + term2 + term3) / N_TOTAL

# ============================================================
# THRESHOLD FINDER
# ============================================================

def find_threshold(all_projs, noise_fn, param_range, name):
    S_vals = np.array([compute_SQ(all_projs, noise_fn, p) for p in param_range])
    crossing = None
    for i in range(len(param_range) - 1):
        s1, s2 = S_vals[i], S_vals[i + 1]
        if (s1 - S_C) * (s2 - S_C) <= 0 and abs(s1 - s2) > 1e-12:
            p1, p2 = param_range[i], param_range[i + 1]
            crossing = p1 + (S_C - s1) * (p2 - p1) / (s2 - s1)
            break
    return crossing, S_vals

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("NOISE MODEL COMPARISON FOR KCBS PROTOCOL")
    print("=" * 60)

    all_vecs, all_projs = build_all_projectors()

    # Verify orthogonality
    print("\nOrthogonality check (adjacent pairs must be ~0):")
    edges = [(0,1),(1,2),(2,3),(3,4),(4,0),
             (0,5),(4,5),(1,6),(2,6),(3,7),(4,7)]
    max_overlap = 0.0
    for (i, j) in edges:
        ov = np.abs(np.vdot(all_vecs[i], all_vecs[j]))
        max_overlap = max(max_overlap, ov)
    print(f"  Max adjacent overlap: {max_overlap:.2e}  (should be ~0)")

    # Verify witness
    W = sum(all_projs[:5])
    eigs = np.linalg.eigvalsh(W)
    print(f"\nWitness eigenvalues (5 KCBS projectors):")
    print(f"  {np.round(eigs, 4)}")
    print(f"  Max eigenvalue: {eigs[-1]:.6f}  (should be √5 = {np.sqrt(5):.6f})")

    # IDEAL CASE
    print("\n" + "=" * 60)
    print("IDEAL CASE (NO NOISE)")
    print("=" * 60)
    S_ideal = compute_SQ(all_projs, apply_depolarizing, 1.0)
    print(f"  S_Q(no noise)      = {S_ideal:.6f}")
    print(f"  Expected from paper = 0.921031")
    print(f"  S_C                = {S_C:.6f}")
    print(f"  Gap                = {S_ideal - S_C:+.6f}")

    if abs(S_ideal - 0.921031) > 1e-4:
        print(f"\n  *** WARNING: Ideal value does not match paper ***")
        print(f"  Difference: {abs(S_ideal - 0.921031):.6f}")

    # Noise models
    results = {}

    print("\n" + "=" * 60)
    print("[1] DEPOLARIZING NOISE")
    print("=" * 60)
    mu_range = np.linspace(0.5, 1.0, 501)
    mu_c, _ = find_threshold(all_projs, apply_depolarizing, mu_range, 'depol')
    if mu_c:
        print(f"  Threshold: μ_c = {mu_c:.4f}")
        print(f"  Expected:  0.9822")
        results['depolarizing'] = mu_c
    else:
        print(f"  No crossing found in range")

    print("\n" + "=" * 60)
    print("[2] AMPLITUDE DAMPING NOISE")
    print("=" * 60)
    gamma_range = np.linspace(0.0, 1.0, 501)
    gamma_c, S_amp = find_threshold(all_projs, apply_amplitude_damping, gamma_range, 'amp')
    if gamma_c:
        print(f"  Threshold: γ_c = {gamma_c:.4f}")
        results['amplitude_damping'] = gamma_c
    else:
        print(f"  No crossing found")
        print(f"  S_Q(γ=0) = {S_amp[0]:.6f}")
        print(f"  S_Q(γ=1) = {S_amp[-1]:.6f}")

    print("\n" + "=" * 60)
    print("[3] DEPHASING NOISE")
    print("=" * 60)
    mu_range = np.linspace(0.0, 1.0, 501)
    mu_c, S_deph = find_threshold(all_projs, apply_dephasing, mu_range, 'deph')
    if mu_c:
        print(f"  Threshold: μ_c = {mu_c:.4f}")
        results['dephasing'] = mu_c
    else:
        print(f"  No crossing found")
        print(f"  S_Q(μ=1) = {S_deph[-1]:.6f}")
        print(f"  S_Q(μ=0) = {S_deph[0]:.6f}")

    print("\n" + "=" * 60)
    print("[4] BIT-FLIP NOISE")
    print("=" * 60)
    mu_range = np.linspace(0.0, 1.0, 501)
    mu_c, S_bf = find_threshold(all_projs, apply_bit_flip, mu_range, 'bf')
    if mu_c:
        print(f"  Threshold: μ_c = {mu_c:.4f}")
        results['bit_flip'] = mu_c
    else:
        print(f"  No crossing found")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Noise Model':<25} {'Threshold':>15}")
    print("-" * 50)
    for name, val in results.items():
        print(f"{name:<25} {val:>15.4f}")

    if not results:
        print("  No thresholds found.")
        print("  This suggests the ideal S_Q is incorrect or")
        print("  the noise is not strong enough in the tested range.")

if __name__ == "__main__":
    main()
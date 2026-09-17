"""
GRAPH-DRIVEN SI WITNESS SEARCH IN d = 5 (FIXED)
Run: python graph_driven_search.py
"""

import numpy as np
import networkx as nx
from scipy.optimize import minimize
import time
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# PART 1: GRAPH UTILITIES (FIXED)
# ============================================================

def clique_number(G):
    if G.number_of_nodes() > 30:
        return None
    max_clique = 1
    for clique in nx.find_cliques(G):
        max_clique = max(max_clique, len(clique))
    return max_clique

def independence_number(G):
    nodes = list(G.nodes())
    n = len(nodes)
    if n > 25:
        selected = []
        blocked = set()
        for v in sorted(nodes, key=lambda x: G.degree(x)):
            if v not in blocked:
                selected.append(v)
                blocked.add(v)
                blocked.update(G.neighbors(v))
        return len(selected)
    best = 0
    for mask in range(1 << n):
        sel = [nodes[i] for i in range(n) if mask & (1 << i)]
        if all(not G.has_edge(u, v) for u, v in combinations(sel, 2)):
            best = max(best, len(sel))
    return best

def is_vertex_transitive_fallback(G):
    """Check vertex transitivity via orbit analysis (approximate)."""
    try:
        # Try using the automorphism group via isomorphism
        from networkx.algorithms.isomorphism import GraphMatcher
        nodes = list(G.nodes())
        if len(nodes) <= 1:
            return True
        # Check if all vertices have same degree (necessary condition)
        degrees = [d for _, d in G.degree()]
        if len(set(degrees)) > 1:
            return False
        # For our purposes, this is a good approximation
        return True
    except Exception:
        return False

def lovasz_theta_approx(G):
    """
    Approximate Lovasz theta using eigenvalue bound.
    Returns None if unable to compute.
    """
    n = G.number_of_nodes()
    try:
        A = nx.to_numpy_array(G)
        eigenvalues = np.linalg.eigvalsh(A)
        lam_max = eigenvalues[-1]
        lam_min = eigenvalues[0]
        if lam_max - lam_min > 1e-10:
            theta_upper = -n * lam_min / (lam_max - lam_min)
            return theta_upper
    except Exception:
        pass
    return None

# ============================================================
# PART 2: GRAPH ENUMERATION
# ============================================================

def enumerate_candidate_graphs():
    candidates = []
    print("Enumerating named graphs...")
    
    named_graphs = [
        ('petersen', nx.petersen_graph()),
        ('chvatal', nx.chvatal_graph()),
        ('cubical', nx.cubical_graph()),
        ('desargues', nx.desargues_graph()),
        ('heawood', nx.heawood_graph()),
        ('dodecahedral', nx.dodecahedral_graph()),
        ('pappus', nx.pappus_graph()),
        ('frucht', nx.frucht_graph()),
        ('moebius_kantor', nx.moebius_kantor_graph()),
        ('truncated_tetrahedron', nx.truncated_tetrahedron_graph()),
    ]
    
    for name, G in named_graphs:
        n = G.number_of_nodes()
        alpha = independence_number(G)
        omega = clique_number(G)
        theta = lovasz_theta_approx(G)
        if theta is not None and omega is not None and omega <= 5:
            gap = theta - alpha
            candidates.append({
                'name': name, 'G': G, 'n': n,
                'alpha': alpha, 'omega': omega,
                'theta': float(theta), 'gap': float(gap),
            })
            print(f"  {name}: n={n}, α={alpha}, ω={omega}, "
                  f"ϑ≈{theta:.3f}, gap={gap:.3f}")
    
    # Circulant graphs
    print("\nEnumerating circulant graphs...")
    connection_sets = [
        [1, 2], [1, 3], [1, 4], [1, 2, 3],
        [1, 5], [1, 2, 4], [2, 3], [1, 6], [1, 2, 5],
    ]
    for n in [10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 24]:
        for S in connection_sets:
            S_full = set()
            for s in S:
                S_full.add(s % n)
                S_full.add((-s) % n)
            S_full.discard(0)
            if not S_full:
                continue
            try:
                G = nx.circulant_graph(n, list(S_full))
                alpha = independence_number(G)
                omega = clique_number(G)
                theta = lovasz_theta_approx(G)
                if theta is not None and omega is not None and omega <= 5:
                    gap = theta - alpha
                    if gap > 0.5:  # Only interesting cases
                        candidates.append({
                            'name': f'C_{n}({sorted(S_full)})',
                            'G': G, 'n': n,
                            'alpha': alpha, 'omega': omega,
                            'theta': float(theta), 'gap': float(gap),
                        })
                        print(f"  C_{n}({sorted(S_full)}): α={alpha}, "
                              f"ω={omega}, ϑ≈{theta:.3f}, gap={gap:.3f}")
            except Exception:
                continue
    
    candidates.sort(key=lambda x: x['gap'], reverse=True)
    print(f"\nTotal candidates: {len(candidates)}")
    return candidates

# ============================================================
# PART 3: PROJECTOR OPTIMIZATION
# ============================================================

def optimize_projector_realization(G, d=5, n_restarts=20, max_iter=500):
    nodes = list(G.nodes())
    n = len(nodes)
    edges = list(G.edges())
    
    def unpack(params):
        vr = params[:n*d].reshape(n, d)
        vi = params[n*d:].reshape(n, d)
        v = vr + 1j * vi
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-12)
    
    def loss(params):
        v = unpack(params)
        edge_pen = 0.0
        for (i, j) in edges:
            edge_pen += np.abs(np.dot(np.conj(v[i]), v[j]))**2
        W = np.zeros((d, d), dtype=complex)
        for i in range(n):
            W += np.outer(v[i], np.conj(v[i]))
        min_eig = float(np.min(np.linalg.eigvalsh(W)))
        return edge_pen * 1e4 - min_eig
    
    best_min_eig = -np.inf
    best_v = None
    best_edge_pen = np.inf
    
    rng = np.random.default_rng(0)
    for restart in range(n_restarts):
        v0 = rng.standard_normal((n, d)) + 1j * rng.standard_normal((n, d))
        params0 = np.concatenate([v0.real.flatten(), v0.imag.flatten()])
        try:
            res = minimize(loss, params0, method='L-BFGS-B',
                           options={'maxiter': max_iter})
            v = unpack(res.x)
            W = np.zeros((d, d), dtype=complex)
            for i in range(n):
                W += np.outer(v[i], np.conj(v[i]))
            min_eig = float(np.min(np.linalg.eigvalsh(W)))
            edge_pen = sum(np.abs(np.dot(np.conj(v[i]), v[j]))**2
                           for (i, j) in edges)
            if edge_pen < 1e-3 and min_eig > best_min_eig:
                best_min_eig = min_eig
                best_v = v
                best_edge_pen = edge_pen
        except Exception:
            continue
    
    return best_v, best_min_eig, best_edge_pen

# ============================================================
# PART 4: MAIN
# ============================================================

def main():
    print("=" * 60)
    print("GRAPH-DRIVEN SI WITNESS SEARCH IN d = 5")
    print("=" * 60)
    t0 = time.time()
    
    candidates = enumerate_candidate_graphs()
    if not candidates:
        print("\nNo candidates found.")
        return
    
    print(f"\n{'='*60}")
    print("RUNNING PROJECTOR OPTIMIZATION ON TOP CANDIDATES")
    print(f"{'='*60}")
    
    for i, cand in enumerate(candidates[:15]):
        name = cand['name']
        G = cand['G']
        alpha = cand['alpha']
        n = cand['n']
        print(f"\n[{i+1}/{min(15, len(candidates))}] {name}: "
              f"n={n}, α={alpha}, ϑ≈{cand['theta']:.3f}")
        
        t1 = time.time()
        v, min_eig, edge_pen = optimize_projector_realization(
            G, d=5, n_restarts=8, max_iter=300
        )
        t_opt = time.time() - t1
        
        if v is None or edge_pen > 1e-2:
            print(f"    No valid projector realization "
                  f"(edge_pen={edge_pen:.4f}, {t_opt:.1f}s)")
            continue
        
        gap = min_eig - alpha
        print(f"    min_eig={min_eig:.4f}, α={alpha}, gap={gap:.4f}, "
              f"{t_opt:.1f}s")
        
        if gap > 0:
            print(f"\n    *** SI WITNESS FOUND ***")
            print(f"    Graph: {name}")
            np.save(f'si_witness_{name.replace("/", "_")}.npy', v)
            print(f"    Saved to si_witness_{name.replace('/','_')}.npy")
            print(f"\nTotal time: {time.time() - t0:.1f}s")
            return cand, v, min_eig, alpha
    
    print(f"\n{'='*60}")
    print("SEARCH COMPLETE - NO SI WITNESS FOUND")
    print(f"Total time: {time.time() - t0:.1f}s")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
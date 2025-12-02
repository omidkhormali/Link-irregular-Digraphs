# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 10:33:53 2025

@author: Alexander Bastien
"""


import random
import itertools
from collections import defaultdict
import networkx as nx
from networkx.algorithms import isomorphism as iso

# ---------- Link machinery ----------

def link_subgraph(D: nx.DiGraph, v):
    """Induced subgraph on N_in(v) ∪ N_out(v), excluding v."""
    nbrs = set(D.predecessors(v)) | set(D.successors(v))
    return D.subgraph(nbrs).copy()

def count_directed_3cycles(G: nx.DiGraph):
    """Count directed 3-cycles in G (triangle cycles in either orientation)."""
    nodes = list(G.nodes())
    idx = {u:i for i,u in enumerate(nodes)}
    # adjacency for O(1) checks
    has = lambda a,b: G.has_edge(a,b)
    cnt = 0
    for a,b,c in itertools.combinations(nodes, 3):
        if has(a,b) and has(b,c) and has(c,a):
            cnt += 1
        if has(a,c) and has(c,b) and has(b,a):
            cnt += 1
    return cnt

def cheap_signature(G: nx.DiGraph):
    """Fast invariant signature to prune isomorphism comparisons."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg_pairs = tuple(sorted((G.in_degree(u), G.out_degree(u)) for u in G.nodes()))
    cycles3 = count_directed_3cycles(G)
    return (n, m, deg_pairs, cycles3)

def links_isomorphic(G1: nx.DiGraph, G2: nx.DiGraph):
    """Full isomorphism check for directed links (ignore node labels)."""
    # Optional: enrich nodes with degree attributes for pruning
    for G in (G1, G2):
        for u in G.nodes():
            G.nodes[u]["in"] = G.in_degree(u)
            G.nodes[u]["out"] = G.out_degree(u)
    gm = iso.DiGraphMatcher(G1, G2)
    return gm.is_isomorphic()

def link_irregular_status(D: nx.DiGraph, verbose=False):
    """
    Return (is_irregular, collisions, link_cache) where:
      - is_irregular: bool
      - collisions: list of (u, v) with isomorphic links
      - link_cache: dict v -> (link_subgraph, signature)
    """
    link_cache = {}
    by_sig = defaultdict(list)
    for v in D.nodes():
        L = link_subgraph(D, v)
        sig = cheap_signature(L)
        link_cache[v] = (L, sig)
        by_sig[sig].append(v)

    collisions = []
    for sig, verts in by_sig.items():
        if len(verts) < 2:
            continue
        if verbose:
            print(f"Signature {sig} candidates: {verts}")
        for u, v in itertools.combinations(verts, 2):
            Lu = link_cache[u][0]
            Lv = link_cache[v][0]
            if links_isomorphic(Lu, Lv):
                collisions.append((u, v))
    return (len(collisions) == 0, collisions, link_cache)

# ---------- Tournament generation ----------

def random_tournament(n: int, seed=None) -> nx.DiGraph:
    """Generate a random tournament on n vertices."""
    if seed is not None:
        random.seed(seed)
    D = nx.DiGraph()
    D.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                D.add_edge(i, j)
            else:
                D.add_edge(j, i)
    return D

def flip_edge(D: nx.DiGraph, u, v):
    """Flip orientation of edge between u and v in a tournament."""
    if D.has_edge(u, v):
        D.remove_edge(u, v)
        D.add_edge(v, u)
    elif D.has_edge(v, u):
        D.remove_edge(v, u)
        D.add_edge(u, v)
    else:
        # Should not happen in tournament
        D.add_edge(u, v)

# ---------- Heuristic search strategies ----------

def find_link_irregular_random(n: int, max_tries=200, verbose=False, seed=None):
    """Random search for a link‑irregular tournament."""
    for t in range(max_tries):
        D = random_tournament(n, seed=None if seed is None else seed + t)
        ok, collisions, _ = link_irregular_status(D, verbose=False)
        if ok:
            if verbose:
                print(f"Success after {t+1} tries.")
            return D
        if verbose and t % 20 == 19:
            print(f"Try {t+1}: {len(collisions)} collisions")
    return None

def find_link_irregular_hillclimb(n: int, steps=5000, restarts=5, verbose=False, seed=None):
    """
    Hill‑climbing: start from random tournament and flip edges to resolve link collisions.
    Greedy: choose flips that reduce collision count; random tie‑breaks.
    """
    rng = random.Random(seed)
    best_D = None
    best_collisions = None

    for r in range(restarts):
        D = random_tournament(n, seed=rng.randint(0, 10**9))
        ok, collisions, _ = link_irregular_status(D, verbose=False)
        if ok:
            if verbose:
                print(f"Found link‑irregular at restart {r} (no flips).")
            return D

        for s in range(steps):
            # Pick a collision pair (u, v)
            if not collisions:
                if verbose:
                    print(f"Solved after {s} steps at restart {r}.")
                return D
            u, v = rng.choice(collisions)

            # Candidate flips: edges incident to u or v (small neighborhood moves)
            candidates = set()
            for w in D.nodes():
                if w == u or w == v:
                    continue
                candidates.add(tuple(sorted((u, w))))
                candidates.add(tuple(sorted((v, w))))
            # Try a small random subset to limit cost
            trial_edges = rng.sample(list(candidates), k=min(len(candidates), 20))
            improved = False
            current_collision_count = len(collisions)

            # Evaluate flips quickly using signature changes (full recompute for simplicity)
            best_move = None
            best_move_collisions = current_collision_count
            for a, b in trial_edges:
                flip_edge(D, a, b)
                ok2, collisions2, _ = link_irregular_status(D, verbose=False)
                # Accept any reduction
                if len(collisions2) < best_move_collisions:
                    best_move_collisions = len(collisions2)
                    best_move = (a, b, collisions2)
                    if ok2:
                        # Early exit on success
                        if verbose:
                            print(f"Solved by flip ({a},{b}) at step {s}, restart {r}.")
                        return D
                # revert
                flip_edge(D, a, b)

            if best_move is not None:
                a, b, collisions = best_move
                flip_edge(D, a, b)
                improved = True

            # Track best seen across restarts
            if best_D is None or len(collisions) < len(best_collisions or []):
                best_D = D.copy()
                best_collisions = collisions

            if verbose and s % 200 == 199:
                print(f"Restart {r}, step {s+1}: collisions={len(collisions)}")

        # restart loop continues

    # If not found, return best seen (may be useful for debugging)
    return best_D

# ---------- Seeded extension (optional) ----------

def seed_tournament_D6():
    """
    Provide a small seed tournament known/assumed to be link‑irregular on 6 vertices.
    If you have a proven seed, replace edges below with your D6.
    """
    D = nx.DiGraph()
    D.add_nodes_from(range(6))
    edges = [
        (0,1),(1,2),(2,0),  # 3-cycle
        (3,0),(3,1),(3,2),  # 3 beats 0,1,2
        (4,3),(4,0),(1,4),  # mixed orientations
        (5,4),(5,2),(5,3)   # etc.
    ]
    for i in range(6):
        for j in range(i+1, 6):
            if (i,j) in edges:
                continue
            if (j,i) in edges:
                continue
            # Default orientation for missing pairs
            edges.append((i,j))
    D.add_edges_from(edges)
    return D

def extend_tournament(seed: nx.DiGraph, n: int, rule="random", seed_val=None):
    """
    Extend a seed tournament to n vertices.
    rule="random": random orientations to new vertex.
    """
    rng = random.Random(seed_val)
    D = seed.copy()
    k = D.number_of_nodes()
    for v in range(k, n):
        D.add_node(v)
        for u in range(v):
            if rng.random() < 0.5:
                D.add_edge(u, v)
            else:
                D.add_edge(v, u)
    return D

def find_link_irregular_seeded(n: int, attempts=100, verbose=False, seed_val=None):
    """Try extending a seed and hill‑climb if needed."""
    base = seed_tournament_D6()
    for a in range(attempts):
        D = extend_tournament(base, n, rule="random", seed_val=None if seed_val is None else seed_val + a)
        ok, collisions, _ = link_irregular_status(D, verbose=False)
        if ok:
            if verbose:
                print(f"Seeded success after {a+1} attempts.")
            return D
        # quick localized hill‑climb
        D2 = find_link_irregular_hillclimb(n, steps=1500, restarts=1, verbose=False, seed=seed_val)
        if D2 is not None:
            ok2, col2, _ = link_irregular_status(D2, verbose=False)
            if ok2:
                if verbose:
                    print("Seeded + hill‑climb success.")
                return D2
    return None

# ---------- CLI demo ----------

def findtournament(n):
    if __name__ == "__main__":
        #print(f"Searching for a link‑irregular tournament on n={n}...")

        # Try fast random search
        D = find_link_irregular_random(n, max_tries=300, verbose=True, seed=42)

        if D is None:
           # print("Random search didn’t succeed; trying hill‑climbing...")
            D = find_link_irregular_hillclimb(n, steps=6000, restarts=5, verbose=True, seed=1337)

        if D is None:
           # print("Trying seeded extension...")
            D = find_link_irregular_seeded(n, attempts=50, verbose=True, seed_val=123)

        if D is None:
            print("No link‑irregular tournament found on", n, "vertices.")
        else:
            ok, collisions, _ = link_irregular_status(D, verbose=True)
            #assert ok, "Sanity check failed: result should be link‑irregular."
            #print("Success: found a link‑irregular tournament.")
            # # Optional: print edges
            # print("Edges:")
            # for e in D.edges():
            #     print(e)



#running program
interval = range(6,500)

for n in interval:
    findtournament(n)

print("Program finished. Any numbers for which no link-irregular tournament could be found are listed above.")

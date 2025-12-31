import numpy as np
import scipy.sparse as sp

def build_A_csc(n, edges):
    """
    Build the column-stochastic link matrix A in sparse CSC form.

    Pages are 1..n.
    Each edge (src, dst) means: src links to dst.

    A_{dst, src} = 1/outdeg(src) if src -> dst exists, else 0.
    """
    outdeg = np.zeros(n + 1, dtype=int)
    for src, dst in edges:
        outdeg[src] += 1

    rows, cols, data = [], [], []
    for src, dst in edges:
        if outdeg[src] == 0:
            continue
        rows.append(dst - 1)          # destination row (0-indexed)
        cols.append(src - 1)          # source column (0-indexed)
        data.append(1.0 / outdeg[src])

    A = sp.csc_matrix((data, (rows, cols)), shape=(n, n))
    dangling = (outdeg[1:] == 0)      # pages with no outgoing links
    return A, dangling, outdeg[1:]

def pagerank_power(A, m=0.15, tol=1e-12, max_iter=500, x0=None, dangling=None):
    """
    Compute x satisfying x = (1-m) A x + m s  (paper eq. (3.2)),
    with standard dangling-node fix:
       dangling_mass = sum_{j dangling} x_j
       A x term is augmented by dangling_mass * s

    If m=0, this reduces to x = A x (paper eq. (2.1)) when there are no dangling nodes.
    """
    n = A.shape[0]
    s = np.ones(n) / n

    if x0 is None:
        x = s.copy()
    else:
        x = np.array(x0, dtype=float)
        x = x / x.sum()

    if dangling is None:
        dangling = np.zeros(n, dtype=bool)

    for k in range(max_iter):
        dangling_mass = x[dangling].sum()
        x_new = (1 - m) * (A @ x + dangling_mass * s) + m * s

        diff = np.abs(x_new - x).sum()  # L1 distance
        x = x_new
        if diff < tol:
            return x, k + 1, diff

    return x, max_iter, diff


def eigenspace_dim_near_1(A_dense, atol=1e-9):
    """For small examples: estimate dim(V1(A)) by counting eigenvalues near 1."""
    w = np.linalg.eigvals(A_dense)
    return int(np.sum(np.isclose(w, 1.0, atol=atol))), w

edges_fig21 = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,1),(4,1),(4,3)]
A21, dang21, _ = build_A_csc(4, edges_fig21)

x_A, it_A, _ = pagerank_power(A21, m=0.0, tol=1e-14, max_iter=2000, dangling=dang21)
x_M, it_M, _ = pagerank_power(A21, m=0.15, tol=1e-14, max_iter=2000, dangling=dang21)

print("Fig2.1 using A (m=0):", x_A, "iters:", it_A)
print("Fig2.1 using M (m=0.15):", x_M, "iters:", it_M)

edges_fig22 = [(1,2),(2,1),(3,4),(4,3),(5,3),(5,4)]
A22, dang22, _ = build_A_csc(5, edges_fig22)

dimV1, eigvals = eigenspace_dim_near_1(A22.toarray())
print("dim(V1(A)) approx:", dimV1, "eigs:", eigvals)

x_M22, it_M22, _ = pagerank_power(A22, m=0.15, tol=1e-14, max_iter=2000, dangling=dang22)
print("Fig2.2 using M (m=0.15):", x_M22, "iters:", it_M22)

# STARTING EXERCISE 1:

edges_ex1 = [
    (1,2),(1,3),(1,4),
    (2,3),(2,4),
    (3,1),(3,5),   # page 3 now links to 1 and 5
    (4,1),(4,3),
    (5,3)          # page 5 links to 3
]
Aex1, dang_ex1, _ = build_A_csc(5, edges_ex1)

x_ex1_A, _, _ = pagerank_power(Aex1, m=0.0, tol=1e-14, max_iter=5000, dangling=dang_ex1)
print("Exercise 1 (using A, m=0):", x_ex1_A)
print("Compare x3 vs x1:", x_ex1_A[2], "vs", x_ex1_A[0])

# STARTING EXERCISE 2:

edges_ex2 = [(1,2),(2,1),(3,4),(4,3),(5,6),(6,5),(5,7),(6,7),(7,6)]
Aex2, dang_ex2, _ = build_A_csc(7, edges_ex2)

dimV1, eigvals = eigenspace_dim_near_1(Aex2.toarray())
print("Exercise 2: dim(V1(A)) approx:", dimV1)
print("eigenvalues:", eigvals)

# STARTING EXERCISE 3:

def build_A_dense(n, edges):
    """
    Dense column-stochastic link matrix A.
    Edge (src,dst) means src links to dst.
    So A[dst-1, src-1] = 1/outdeg(src).
    """
    outdeg = np.zeros(n + 1, dtype=int)
    for src, dst in edges:
        outdeg[src] += 1

    A = np.zeros((n, n), dtype=float)
    for src, dst in edges:
        A[dst - 1, src - 1] += 1.0 / outdeg[src]
    return A

def dim_V1(A, atol=1e-10):
    eigvals = np.linalg.eigvals(A)
    dim = int(np.sum(np.isclose(eigvals, 1.0, atol=atol)))
    return dim, eigvals

# Figure 2.2 edges:
# 1 -> 2, 2 -> 1, 3 -> 4, 4 -> 3, 5 -> 3 and 4
edges_fig22 = [(1,2),(2,1),(3,4),(4,3),(5,3),(5,4)]

# Exercise 3 modification: add link 5 -> 1
edges_ex3 = edges_fig22 + [(5,1)]

A_ex3 = build_A_dense(5, edges_ex3)

dim, eigvals = dim_V1(A_ex3)
print("dim(V1(A)) =", dim)
print("eigenvalues =", np.sort(eigvals))
print("A =\n", A_ex3)

# STARTING EXERCISE 4:

def build_substochastic_A(n, edges):
    """
    Build the (possibly) column-substochastic link matrix A for a web with dangling nodes.
    Pages are 1..n
    Edge (src,dst) means src links to dst.
    A[dst-1, src-1] = 1/outdeg(src).
    If outdeg(src)=0, that column stays all zeros (dangling node).
    """
    outdeg = np.zeros(n + 1, dtype=int)
    for src, dst in edges:
        outdeg[src] += 1

    A = np.zeros((n, n), dtype=float)
    for src, dst in edges:
        A[dst - 1, src - 1] += 1.0 / outdeg[src]

    dangling = np.where(outdeg[1:] == 0)[0] + 1  # page numbers
    return A, dangling

# Figure 2.1 edges
# 1 -> {2,3,4}, 2 -> {3,4}, 3 -> {1}, 4 -> {1,3}
edges_fig21 = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,1),(4,1),(4,3)]

# Exercise 4: remove link 3 -> 1  (so page 3 becomes dangling)
edges_ex4 = [e for e in edges_fig21 if e != (3,1)]

A, dangling_pages = build_substochastic_A(4, edges_ex4)
print("Dangling pages:", dangling_pages)
print("A=\n", A)
print("Column sums:", A.sum(axis=0))

# Perron eigenvalue/eigenvector
eigvals, eigvecs = np.linalg.eig(A)
# pick eigenvalue with largest real part (Perron eigenvalue here)
k = np.argmax(eigvals.real)
lam = eigvals[k].real
v = eigvecs[:, k].real

# make it nonnegative and normalize to sum to 1
v = np.abs(v)
x = v / v.sum()

print("\nPerron eigenvalue lambda =", lam)
print("Normalized Perron eigenvector x =", x)
print("sum(x) =", x.sum())

# STARTING EXERCISE 11:

# Augmented Figure 2.1
n = 5
edges = [
    (1,2),(1,3),(1,4),     # page 1
    (2,3),(2,4),           # page 2
    (3,1),(3,5),           # page 3
    (4,1),(4,3),           # page 4
    (5,3)                  # page 5
]

A = build_A_dense(n, edges)

# Google matrix
m = 0.15
S = np.ones((n, n)) / n
M = (1 - m) * A + m * S

# Eigenvector for eigenvalue 1
eigvals, eigvecs = np.linalg.eig(M)
k = np.argmin(np.abs(eigvals - 1))
x = eigvecs[:, k].real
x = x / x.sum()

print("PageRank vector:", x)
print("Sum:", x.sum())

# STARTING EXERCISE 12:

def eigvec_lambda1(M):
    """Eigenvector for eigenvalue 1, normalized to sum to 1 (positive direction)."""
    w, V = np.linalg.eig(M)
    k = np.argmin(np.abs(w - 1))
    x = V[:, k].real
    if x.sum() < 0:
        x = -x
    x = x / x.sum()
    return x

def ranking(x):
    order = np.argsort(-x)
    return [(int(i + 1), float(x[i])) for i in order]

# Exercise 11 web: Figure 2.1 + page 5 with 3<->5
# 1 -> {2,3,4}
# 2 -> {3,4}
# 3 -> {1,5}
# 4 -> {1,3}
# 5 -> {3}
edges_5 = [
    (1,2),(1,3),(1,4),
    (2,3),(2,4),
    (3,1),(3,5),
    (4,1),(4,3),
    (5,3)
]

# Exercise 12: add page 6 linking to every other page (1..5), no one links to 6
edges_6 = edges_5 + [(6,i) for i in range(1,6)]

n = 6
A = build_A_dense(n, edges_6)

# Rank using A (importance scores x = A x)
xA = eigvec_lambda1(A)

# Rank using M = (1-m)A + mS, with m=0.15
m = 0.15
S = np.ones((n, n)) / n
M = (1 - m) * A + m * S
xM = eigvec_lambda1(M)

print("x using A:", xA)
print("ranking(A):", ranking(xA))
print("x using M:", xM)
print("ranking(M):", ranking(xM))

# STARTING EXERCISE 13:

def pagerank_google(A, m=0.15):
    n = A.shape[0]
    S = np.ones((n, n)) / n
    M = (1 - m) * A + m * S

    w, V = np.linalg.eig(M)
    k = np.argmin(np.abs(w - 1))
    x = V[:, k].real
    if x.sum() < 0:
        x = -x
    x = x / x.sum()
    return x

# Exercise 13 web (two subwebs)
n = 6
edges = [
    # Subweb A: {1,2,3}
    (1,2),(2,3),(3,1),(1,3),(3,2),

    # Subweb B: {4,5}
    (4,5),(5,4),

    # Page 6 points into both subwebs, no one links to 6
    (6,1),(6,4)
]

A = build_A_dense(n, edges)
x = pagerank_google(A, m=0.15)

ranking = sorted([(i+1, x[i]) for i in range(n)], key=lambda t: -t[1])
print("PageRank vector:", x)
print("Ranking (page, score):")
for p, s in ranking:
    print(p, float(s))

# STARTING EXERCISE 14:

n = 5
edges = [
    (1,2),(1,3),(1,4),
    (2,3),(2,4),
    (3,1),(3,5),
    (4,1),(4,3),
    (5,3)
]

outdeg = np.zeros(n+1, dtype=int)
for s,d in edges:
    outdeg[s] += 1

A = np.zeros((n,n), float)
for s,d in edges:
    A[d-1, s-1] += 1.0/outdeg[s]

m = 0.15
S = np.ones((n,n))/n
M = (1-m)*A + m*S

# steady-state eigenvector q for eigenvalue 1
eigvals, eigvecs = np.linalg.eig(M)
k = np.argmin(np.abs(eigvals - 1))
q = eigvecs[:,k].real
if q.sum() < 0:
    q = -q
q = q / q.sum()

# pick an x0 "not too close" to q: all mass on page 1
x0 = np.zeros(n); x0[0] = 1.0

def l1(v): return float(np.sum(np.abs(v)))

def Mkx(k):
    x = x0.copy()
    for _ in range(k):
        x = M @ x
    return x

# requested k values
Ks = [1,5,10,50]

# errors and ratios
err0 = l1(x0 - q)
results = []
for k in Ks:
    xk = Mkx(k)
    errk = l1(xk - q)
    if k == 1:
        ratio = errk / err0
    else:
        ratio = errk / l1(Mkx(k-1) - q)
    results.append((k, errk, ratio))

# c = max_j |1 - 2 min_i M_ij|
mins = M.min(axis=0)
c = float(np.max(np.abs(1 - 2*mins)))

# |lambda2|
abs_sorted = sorted(np.abs(eigvals), reverse=True)
lambda2_abs = abs_sorted[1]

print("q =", q)
print("results (k, ||M^k x0 - q||_1, ratio):")
for row in results:
    print(row)
print("c =", c)
print("|lambda2| =", lambda2_abs)

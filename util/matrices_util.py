import numpy as np
import gudhi
import os
from scipy.optimize import minimize
from itertools import combinations


def f(x):
    return (x * (np.log(x + (x < 0) * (-x) + 1e-12) - 1)).sum(axis=1)


def f_grad(x):
    x = x + (x < 0) * (-x)
    return np.log(x + 1e-12)


def k_l(x, y):
    x = x + (x < 0) * (-x)
    y = y + (y < 0) * (-y)

    return (x * np.log(x / (y + 1e-12) + 1e-12) - x + y).sum(axis=1)


def circum_ball(Q, f, f_grad, method='BFGS', verbose=False):
    if Q.shape[0] == 1:
        return Q[0], 0
    f_Q = f(Q)
    hull = np.concatenate((Q, f_Q[:, None]), axis=1)
    x0 = np.full(hull.shape[0] - 1, 1 / hull.shape[0])

    F_opt = lambda x: f((hull[1:].T @ x + (1 - x.sum()) * hull[0])[None, 0:12]) - (
            hull[1:, 12].T @ x + (1 - x.sum()) * hull[0, 12])

    if verbose:
        print('F_opt in x0:', F_opt(x0))
    F_opt_grad = lambda x: ((hull[1:, 0:12] - hull[0, 0:12]) @ f_grad((hull[1:].T @ x + (1 - x.sum()) *
                                                                       hull[0])[None, 0:12]).T - (hull[1:, 12] -
                                                                                                  hull[0, 12])[:,
                                                                                                 None]).flatten()

    z = minimize(fun=F_opt, x0=x0, method=method, jac=F_opt_grad)

    if verbose:
        print(z)

    q = (hull[1:].T @ z.x + (1 - z.x.sum()) * hull[0])[0:12]
    r = -z.fun

    return q, r


def cech_radius(arr, k):
    n = len(arr)

    marked = {}
    for i in range(1, k + 1):
        for P in list(combinations(range(n), i)):
            if P not in marked.keys():
                q, r = circum_ball(arr[P, :], f, f_grad, 'BFGS')
                marked[P] = (q, r)
            for a in range(n):
                if a not in P:
                    if k_l(arr[a: a + 1], q) < r:
                        marked[tuple(sorted(list(P) + [a]))] = (q, r)

    return marked


def build_filtration(marked, compute_persistence=True):
    st = gudhi.SimplexTree()
    for simplex, val in marked.items():
        st.insert(sorted(list(simplex)), filtration=val[1])

    if compute_persistence:
        st.compute_persistence()
    return st


def get_statistics(pers_hom):
    if len(pers_hom) == 0:
        return [0, 0, 0]

    ph = pers_hom[np.where(np.isfinite(pers_hom[:, 1]))]
    l = ph[:, 1] - ph[:, 0]

    if len(l) == 0:
        mean = 0
        std = 0
    else:
        mean = np.mean(l)
        std = np.std(l)

    m = np.max(ph[:, 1])
    ph_ = np.nan_to_num(pers_hom, posinf=m + 1)

    l_ = ph_[:, 1] - ph_[:, 0]
    p = l_ / l_.sum()

    entropy = -(np.log(p + 1e-12) * p).sum()

    return [mean, std, entropy]


def transform_matrix(matrix):
    modified_matrix = np.zeros_like(matrix, dtype=float)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                modified_matrix[i, j] = 1 / matrix[i, j]
    return modified_matrix


def process_matrices(mat_folder, max_homology_dim=2, columns_as_vertices=False, rips=False):
    arr_weighted_matrix = []
    for i in range(4):
        matrix = np.loadtxt(os.path.join(mat_folder, f"{i}.txt"))
        if rips:
            matrix = transform_matrix(matrix)
        if columns_as_vertices:
            matrix = matrix.T
        arr_weighted_matrix.append(matrix)
    stats = []
    phs = []
    for i in range(4):
        marked = cech_radius(arr_weighted_matrix[i], max_homology_dim + 1)
        st = build_filtration(marked)
        for j in range(max_homology_dim):
            ph = st.persistence_intervals_in_dimension(j)
            stats += get_statistics(ph)
            phs.append(ph)

    return stats, phs

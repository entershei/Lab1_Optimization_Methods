import numpy as np

from .simplex_error import *


def solve_canonical(c, A, b):
    """Solve the linear program of form:

    Maximize f(x) = cᵀx

    Subject to Ax = b, x ≥ 0

    """

    initial_basis = solve_aux_program(A, b)
    basis, Q = solve_canonical_impl(initial_basis, c, A, b)
    x = np.zeros(len(c), dtype="float64")
    for i, j in enumerate1(basis):
        assert Q[i][j] == 1
        x[j - 1] = Q[i][0]
    return x


def solve_canonical_impl(basis, c, A, b):
    """Solve a linear program in canonical form with a given basis."""
    (m, n) = A.shape
    Q = np.row_stack(
        (
            np.hstack(([0], -c)),
            np.column_stack((b, A)),
        )
    )
    gauss_elimination(Q, basis)

    while True:
        # choose 's' and 'r' according to the Bland's rule
        ss = (j for j in range(1, n + 1) if Q[0][j] < 0)
        s = min(ss, default=None)
        if s is None:
            return basis, Q

        rs = [i for i in range(1, m + 1) if Q[i][s] > 0]  # and Q[0][s] / Q[i][s] > 0
        r = min(rs, key=lambda i: (abs(Q[0][s] / Q[i][s]), basis[i - 1]), default=None)
        if r is None:
            raise UnboundFunction

        Q[r] /= Q[r][s]
        for i in range(m + 1):
            if i != r:
                Q[i] -= Q[r] * Q[i][s]

        basis[r - 1] = s


def solve_aux_program(A, b):
    """Solve the auxilliary program to find an initial basis for the
    original system.

    The original system is the one subject to Ax = b, x ≥ 0. The
    auxilliary program introduces a set of new variables 'y', |y| =
    |b| and maximizes g(x, y) = -∑y, subject to Ax + y = b, x ≥ 0,
    y ≥ 0. The benefit is that the initial basis for this
    system is trivial, it's just y.

    It's obvious, that the optimal solution is y = 0 and x = some
    solution for the original system if it exists. It should have some
    basis in x.

    Returns that very basis.

    """

    (m, n) = A.shape  # m = |b| = |y|, n = |x|

    c_ = np.zeros(n + m, dtype="float64")
    c_[n:] = -1  # g = -∑y

    A_ = np.hstack((A, np.eye(m)))  # = Ax + y
    b_ = b.copy()
    for i, b_i in enumerate(b):
        if b_i < 0:
            b_[i] *= -1
            A_[i] *= -1

    initial_basis = list(range(n + 1, n + m + 1))

    basis, Q = solve_canonical_impl(initial_basis, c_, A_, b_)
    if Q[0][0] < 0:
        raise NoSolutionError("The original system is inconsistent")

    redundant = {}
    for no, s in enumerate(basis):
        if s <= n:
            continue
        r = next(i for i in range(1, m + 1) if Q[i][s] == 1)
        k = next((j for j in range(1, n + 1) if Q[r][j] != 0), None)
        if k is None:
            del A[r - 1]
            del Q[r]
            redundant.add(s)
        else:
            basis[no] = k
            gauss_elimination(Q, basis)
    basis = [s for s in basis if s not in redundant]

    return basis


def gauss_elimination(Q, basis):
    (m, n) = Q.shape
    for i, j in enumerate1(basis):
        Q[i] /= Q[i][j]
        for k in range(m):
            if k != i:
                Q[k] -= Q[i] * Q[k][j]


def enumerate1(iterable):
    i = 1
    for el in iterable:
        yield (i, el)
        i += 1

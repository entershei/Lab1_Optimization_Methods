import numpy as np
from random import uniform
from functools import partial


def generate_eigenvalues(n, k):
    eigenvalues = [1, k]
    for i in range(0, n - 2):
        eigenvalues.append(uniform(1, k))
    return eigenvalues


def generate_orthonormal_matrix(n):
    mx = []
    for i in range(0, n):
        mx.append([])
    for i in range(0, n):
        for j in range(0, n):
            mx[i].append(uniform(-100, 100))
    q, r = np.linalg.qr(mx)
    return q


def generate_hessian(n, k):
    eigenvalues = generate_eigenvalues(n, k)
    D = np.diag(eigenvalues)
    A = generate_orthonormal_matrix(n)
    H = np.matmul(np.matmul(A, D), np.transpose(A))
    return H


def fn_from_hessian(hessian, n):
    def fn(x):
        result = 0
        for i in range(0, n):
            for j in range(i, n):
                result = result + hessian[i][j] * x[i] * x[j]
        return result

    return fn


def grad_from_hessian(hessian, n):
    result = []

    def gr(it, x):
        return sum(np.array(x) * np.array(hessian[it]) * [2 if j == it else 1 for j in range(n)])

    for i in range(n):
        result.append(partial(gr, i))
    return result


def generate_quadratic_assignment(n, k):
    hess = generate_hessian(n, k)
    return fn_from_hessian(hess, n), grad_from_hessian(hess, n)

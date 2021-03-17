import numpy as np
# import numdifftools.nd_algopy as nda

from methopt.quadratic_assignment import (
    generate_hessian,
    fn_from_hessian,
    grad_from_hessian
)

EPS = 1e-7


def approx_equal(a, b):
    return abs(a - b) < EPS


def internal_test_hessian(n, k):
    hess = generate_hessian(n, k)
    assert approx_equal(np.linalg.cond(hess), k)
    assert hess.shape == (n, n)


def test_generate_hessian1():
    internal_test_hessian(2, 666)


def test_generate_hessian2():
    internal_test_hessian(100, 3)

#
# def internal_test_fn_from_hessian1(n, k):
#     hess = generate_hessian(n, k)
#     fn = fn_from_hessian(hess)
#     hess2 = nda.Hessian(fn)
#     assert np.linalg.cond(hess2([0] * n)) == k
#
#
# def test_fn_generation1():
#     internal_test_fn_from_hessian1(2, 666)
#
#
# def test_fn_generation2():
#     internal_test_fn_from_hessian1(1000, 6)


def test_fn_from_hessian():
    hess = [
        [10, 12, 11],
        [12, 14, 18],
        [11, 18,  7]
    ]
    fn = fn_from_hessian(hess, 3)
    assert fn([1, 0, 0]) == 10
    assert fn([0, 1, 0]) == 14
    assert fn([0, 0, 1]) == 7

    assert fn([1, 1, 0]) == 10 + 14 + 12
    assert fn([1, 6, 1]) == 10 + 14 * 36 + 7 + 12 * 6 + 18 * 6 + 11


def test_grad_from_hessian():
    hess = [
        [10, 12, 11],
        [12, 14, 18],
        [11, 18,  7]
    ]
    grad = grad_from_hessian(hess, 3)
    assert grad[0]([1, 0, 0]) == 10 * 2
    assert grad[1]([0, 1, 0]) == 14 * 2
    assert grad[2]([0, 0, 1]) == 7 * 2

    assert grad[0]([5, 666, 0]) == 2 * 10 * 5 + 12 * 666

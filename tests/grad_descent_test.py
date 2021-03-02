from optmeth.grad_descent import grad_descent

EPS = 1e-7


def approx_equal(a, b):
    return abs(a - b) < EPS


def test_grad_descent_parabole():
    f = lambda x: (x - 3) ** 2 + 8
    f_grad = lambda x: 2 * (x - 3)
    x0 = -6

    assert approx_equal(grad_descent(f, f_grad, x0), 3)

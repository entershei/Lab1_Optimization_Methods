import numpy as np

from methopt.newtons_method import newtons_method

EPS = 1e-3


def approx_equal(a, b, eps=EPS):
    return np.all(abs(a - b) < eps)


def test1():
    # f = x^2
    H = [[lambda x: 2]]
    grad = [lambda x: 2 * x]
    x0 = [3]
    f = lambda x: x ** 2

    res = newtons_method(f, np.array(H), np.array(grad), np.array(x0))
    assert approx_equal(res, [0])


def test2():
    # f = 10x^2 - 100x
    f = lambda x: 10 * x ** 2 - 100 * x
    Q = [[lambda x: 20]]
    b = [lambda x: -100]
    x0 = [0]

    res = newtons_method(f, np.array(Q), np.array(b), np.array(x0))
    assert approx_equal(res, [5])


def test3():
    # f = x^4 + x^2 + 6x
    H = [[lambda x: 4 * 3 * x ** 2 + 2]]
    grad = [lambda x: 4 * x ** 3 + 2 * x + 6]
    x0 = [3]
    f = lambda x: x ** 4 + x ** 2 + 6 * x

    res = newtons_method(f, np.array(H), np.array(grad), np.array(x0))
    assert approx_equal(res, [-1])


def test4():
    # f = 4x^6 + 10z^2 - 4xz + 10z
    H = [
        [lambda x: 120 * x[0] ** 4, lambda x: -4],
        [lambda x: -4, lambda x: 20]
    ]
    grad = [lambda x: 24 * x[0] ** 5 - 4 * x[1],
            lambda x: 20 * x[1] - 4 * x[0] + 10]
    x0 = [3, 3]
    f = lambda x: 4 * x[0] ** 6 + 10 * x[1] ** 2 - 4 * x[0] * x[1] + 10 * x[1]

    res = newtons_method(f, np.array(H), np.array(grad), np.array(x0))
    assert approx_equal(res, [-0.636601, -0.62732])


def test_iteration_callback():
    # f = 4x^6 + 10z^2 - 4xz + 10z
    H = [
        [lambda x: 120 * x[0] ** 4, lambda x: -4],
        [lambda x: -4, lambda x: 20]
    ]
    grad = [lambda x: 24 * x[0] ** 5 - 4 * x[1],
            lambda x: 20 * x[1] - 4 * x[0] + 10]
    x0 = [3, 3]
    f = lambda x: 4 * x[0] ** 6 + 10 * x[1] ** 2 - 4 * x[0] * x[1] + 10 * x[1]

    trajectory = []
    iteration_callback = lambda x, **kwargs: trajectory.append((x, f(x)))

    newtons_method(
        f,
        np.array(H),
        np.array(grad),
        np.array(x0),
        iteration_callback=iteration_callback
    )

    first_x, first_fx = trajectory[0]
    last_x, last_fx = trajectory[-1]

    assert approx_equal(first_x, [3, 3])
    assert approx_equal(first_fx, 3000)
    assert approx_equal(last_x, [-0.636601, -0.62732])
    assert approx_equal(last_fx, -3.66907)

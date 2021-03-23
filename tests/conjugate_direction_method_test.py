import numpy as np

from methopt.conjugate_direction_method import conjugate_direction_method

EPS = 1e-3


def approx_equal(a, b, eps=EPS):
    return np.all(abs(a - b) < eps)


def test1():
    # f = x^2
    f = lambda x: x[0] ** 2
    grad = lambda x: np.array([2 * x[0]], dtype=np.float64)
    x0 = [-6]

    res = conjugate_direction_method(f, grad, np.array(x0, dtype=np.float64),
                                     max_iterations_count=4, eps=EPS)
    assert approx_equal(res, [0])


def test2():
    # f = 10x^2 - 1000x
    f = lambda x: 10 * x[0] ** 2 - 1000 * x[0]
    grad = lambda x: np.array([20 * x[0] - 1000], dtype=np.float64)
    x0 = [30]

    res = conjugate_direction_method(f, grad, np.array(x0, dtype=np.float64),
                                     max_iterations_count=4, eps=EPS)
    assert approx_equal(res, [50])


def test3():
    # f = x^2 + z^2
    f = lambda x: x[0] ** 2 + x[1] ** 2
    grad = lambda x: np.array([2 * x[0], 2 * x[1]])
    x0 = [10, 10]

    res = conjugate_direction_method(f, grad, np.array(x0), eps=EPS)
    assert approx_equal(res, [0, 0])


def test4():
    # f = x^2 + z^2 + x
    f = lambda x: x[0] ** 2 + x[1] ** 2 + x[0]
    grad = lambda x: np.array([2 * x[0] + 1, 2 * x[1]])
    x0 = [10, 10]

    res = conjugate_direction_method(f, grad, np.array(x0), eps=EPS)
    assert approx_equal(res, [-0.5, 0])


def test5():
    # f = 2x^2 + z^2 + x
    f = lambda x: 2 * x[0] ** 2 + x[1] ** 2 + x[0]
    grad = lambda x: np.array([4 * x[0] + 1, 2 * x[1]], dtype=np.float64)
    x0 = [10, 10]

    res = conjugate_direction_method(f, grad, np.array(x0, dtype=np.float64))
    assert approx_equal(res, [-0.25, 0])



def test6():
    # f = (x^2 - z)^2 + (x - 1)^2
    f = lambda x: (x[0] ** 2 - x[1]) ** 2 + (x[0] - 1) ** 2
    grad = lambda x: np.array(
        [2 * (x[0] ** 2 - x[1]) * 2 * x[0] + 2 * (x[0] - 1),
         -2 * (x[0] ** 2 - x[1])])
    x0 = [-1, -2]

    res = conjugate_direction_method(f, grad, np.array(x0), eps=1e-3,
                                     max_iterations_count=1000)
    assert approx_equal(res, [1, 1])


def test7():
    # f = (x^2 + z^2)^2 + z^3
    f = lambda x: (x[0] ** 2 + x[1] ** 2) ** 2 + x[1] ** 3
    grad = lambda x: np.array(
        [2 * (x[0] ** 2 + x[1] ** 2) * 2 * x[0],
         2 * (x[0] ** 2 + x[1] ** 2) * 2 * x[1] + 3 * x[1] ** 2])
    x0 = [-1.5, -1.5]

    res = conjugate_direction_method(f, grad, np.array(x0), eps=1e-3,
                                     max_iterations_count=1000)
    assert approx_equal(res, [0, -0.75])


def test_iteration_callback():
    # f = x^2 + 5z^2 + 4xz + x
    grad = lambda x: np.array([2 * x[0] + 4 * x[1] + 1, 10 * x[1] + 4 * x[0]])
    x0 = [0, 0]

    f = lambda x: x[0] ** 2 + 5 * x[1] ** 2 + 4 * x[0] * x[1] + x[0]

    trajectory = []
    iteration_callback = lambda x, **kwargs: trajectory.append((x, f(x)))

    conjugate_direction_method(
        f,
        grad,
        np.array(x0),
        iteration_callback=iteration_callback,
        eps=1e-5
    )

    first_x, first_fx = trajectory[0]
    last_x, last_fx = trajectory[-1]

    assert approx_equal(first_x, [0, 0])
    assert approx_equal(first_fx, 0)
    assert approx_equal(last_x, [-2.5, 1])
    assert approx_equal(last_fx, -1.25)

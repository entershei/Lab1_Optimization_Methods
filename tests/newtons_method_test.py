import numpy as np

from methopt.newtons_method import newtons_method

EPS = 1e-3


def approx_equal(a, b, eps=EPS):
    return np.all(abs(a - b) < eps)


def test1():
    # f = x^2
    f = lambda x: x[0] ** 2
    H = lambda x: np.array([[2]])
    grad = lambda x: np.array([2 * x[0]])
    x0 = np.array([3])

    res = newtons_method(f, H, grad, x0)
    assert approx_equal(res, [0])


def test2():
    # f = 10x^2 - 100x
    f = lambda x: 10 * x[0] ** 2 - 100 * x[0]
    H = lambda x: np.array([[20]])
    grad = lambda x: np.array([20 * x[0] - 100])
    x0 = np.array([0])

    res = newtons_method(f, H, grad, x0)
    assert approx_equal(res, [5])


def test3():
    # f = x^4 + x^2 + 6x
    f = lambda x: x[0] ** 4 + x[0] ** 2 + 6 * x[0]
    H = lambda x: np.array([[4 * 3 * x[0] ** 2 + 2]])
    grad = lambda x: np.array([4 * x[0] ** 3 + 2 * x[0] + 6])
    x0 = np.array([3])

    res = newtons_method(f, H, grad, x0)
    assert approx_equal(res, [-1])


def test4():
    # f = 4x^6 + 10z^2 - 4xz + 10z
    f = lambda x: 4 * x[0] ** 6 + 10 * x[1] ** 2 - 4 * x[0] * x[1] + 10 * x[1]
    H = lambda x: np.array([
        [120 * x[0] ** 4, -4],
        [-4, 20],
    ], dtype='float64')
    grad = lambda x: np.array([
        24 * x[0] ** 5 - 4 * x[1],
        20 * x[1] - 4 * x[0] + 10,
    ], dtype='float64')
    x0 = np.array([3, 3], dtype='float64')

    res = newtons_method(f, H, grad, x0)
    assert approx_equal(res, [-0.636601, -0.62732])


def test5():
    # f = (x^2 + z^2)^2 + z^3
    f = lambda x: (x[0] ** 2 + x[1] ** 2) ** 2 + x[1] ** 3
    H = lambda x: np.array([
        [12 * x[0] ** 2 + 4 * x[1] ** 2, 8 * x[1] * x[0]],
        [8 * x[1] * x[0], 4 * x[0] ** 2 + 12 * x[1] ** 2 + 6 * x[1]]
    ])
    grad = lambda x: np.array(
        [2 * (x[0] ** 2 + x[1] ** 2) * 2 * x[0],
         2 * (x[0] ** 2 + x[1] ** 2) * 2 * x[1] + 3 * x[1] ** 2])
    x0 = np.array([-1.5, -1.5])

    res = newtons_method(f, H, grad, x0)
    assert approx_equal(res, [0, -0.75])


def test_iteration_callback():
    # f = 4x^6 + 10z^2 - 4xz + 10z
    f = lambda x: 4 * x[0] ** 6 + 10 * x[1] ** 2 - 4 * x[0] * x[1] + 10 * x[1]
    H = lambda x: np.array([
        [120 * x[0] ** 4, -4],
        [-4, 20],
    ], dtype='float64')
    grad = lambda x: np.array([
        24 * x[0] ** 5 - 4 * x[1],
        20 * x[1] - 4 * x[0] + 10,
    ], dtype='float64')
    x0 = np.array([3, 3], dtype='float64')

    trajectory = []
    iteration_callback = lambda x, **kwargs: trajectory.append((x, f(x)))

    newtons_method(f, H, grad, x0,
                   iteration_callback=iteration_callback)

    first_x, first_fx = trajectory[0]
    last_x, last_fx = trajectory[-1]

    assert approx_equal(first_x, [3, 3])
    assert approx_equal(first_fx, 3000)
    assert approx_equal(last_x, [-0.636601, -0.62732])
    assert approx_equal(last_fx, -3.66907)

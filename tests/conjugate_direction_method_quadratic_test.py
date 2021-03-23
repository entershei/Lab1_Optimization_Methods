import numpy as np

from methopt.conjugate_direction_method import conjugate_direction_method_for_quadratic

EPS = 1e-7


def approx_equal(a, b, eps=EPS):
    return np.all(abs(a - b) < eps)


def test1():
    # f = x^2
    Q = [[2]]
    b = [0]
    x0 = [-6]

    res = conjugate_direction_method_for_quadratic(np.array(Q), np.array(b), np.array(x0))
    print(res)
    assert approx_equal(res, [0])


def test2():
    # f = x^2 + 20x
    Q = [[2]]
    b = [20]
    x0 = [0]

    res = conjugate_direction_method_for_quadratic(np.array(Q), np.array(b), np.array(x0))
    print(res)
    assert approx_equal(res, [-10])


def test2():
    # f = 10x^2 - 100x
    Q = [[20]]
    b = [-100]
    x0 = [0]

    res = conjugate_direction_method_for_quadratic(np.array(Q), np.array(b), np.array(x0))
    print(res)
    assert approx_equal(res, [5])


def test3():
    # f = x^2 + z^2
    Q = [
        [2, 0],
        [0, 2],
    ]
    b = [0, 0]
    x0 = [10, 10]

    res = conjugate_direction_method_for_quadratic(np.array(Q), np.array(b), np.array(x0))
    print(res)
    assert approx_equal(res, [0, 0])


def test4():
    # f = x^2 + z^2 + x
    Q = [
        [2, 0],
        [0, 2],
    ]
    b = [1, 0]
    x0 = [10, 10]

    res = conjugate_direction_method_for_quadratic(np.array(Q), np.array(b), np.array(x0))
    print(res)
    assert approx_equal(res, [-0.5, 0])


def test5():
    # f = 2x^2 + z^2 + x
    Q = [
        [4, 0],
        [0, 2],
    ]
    b = [1, 0]
    x0 = [10, 10]

    res = conjugate_direction_method_for_quadratic(np.array(Q), np.array(b), np.array(x0))
    print(res)
    assert approx_equal(res, [-0.25, 0])


def test6():
    # f = x^2 + 5z^2 + 4xz + x
    Q = [
        [2, 4],
        [4, 10],
    ]
    b = [1, 0]
    x0 = [10, 10]

    res = conjugate_direction_method_for_quadratic(np.array(Q), np.array(b), np.array(x0))
    print(res)
    assert approx_equal(res, [-2.5, 1])


def test_iteration_callback():
    # f = x^2 + 5z^2 + 4xz + x
    Q = [
        [2, 4],
        [4, 10],
    ]
    b = [1, 0]
    x0 = [0, 0]

    f = lambda x: x[0] ** 2 + 5 * x[1] ** 2 + 4 * x[0] * x[1] + x[0]

    trajectory = []
    iteration_callback = lambda x, **kwargs: trajectory.append((x, f(x)))

    conjugate_direction_method_for_quadratic(
        np.array(Q),
        np.array(b),
        np.array(x0),
        iteration_callback=iteration_callback,
    )

    first_x, first_fx = trajectory[0]
    last_x, last_fx = trajectory[-1]

    assert approx_equal(first_x, [0, 0])
    assert approx_equal(first_fx, 0)
    assert approx_equal(last_x, [-2.5, 1])
    assert approx_equal(last_fx, -1.25)

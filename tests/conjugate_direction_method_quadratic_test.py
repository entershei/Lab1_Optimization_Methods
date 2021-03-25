import numpy as np

from methopt.conjugate_direction_method import conjugate_direction_method_for_quadratic
from methopt.utils import TrajectoryIterationCallback

EPS = 1e-7


def approx_equal(a, b, eps=EPS):
    return np.all(abs(a - b) < eps)


def test1():
    # f = x^2
    Q = [[2]]
    b = [0]
    x0 = [-6]

    res = conjugate_direction_method_for_quadratic(
        np.array(Q), np.array(b), np.array(x0)
    )
    print(res)
    assert approx_equal(res, [0])


def test2():
    # f = x^2 + 20x
    Q = [[2]]
    b = [20]
    x0 = [0]

    res = conjugate_direction_method_for_quadratic(
        np.array(Q), np.array(b), np.array(x0)
    )
    print(res)
    assert approx_equal(res, [-10])


def test2():
    # f = 10x^2 - 100x
    Q = [[20]]
    b = [-100]
    x0 = [0]

    res = conjugate_direction_method_for_quadratic(
        np.array(Q), np.array(b), np.array(x0)
    )
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

    res = conjugate_direction_method_for_quadratic(
        np.array(Q), np.array(b), np.array(x0)
    )
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

    res = conjugate_direction_method_for_quadratic(
        np.array(Q), np.array(b), np.array(x0)
    )
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

    res = conjugate_direction_method_for_quadratic(
        np.array(Q), np.array(b), np.array(x0)
    )
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

    res = conjugate_direction_method_for_quadratic(
        np.array(Q), np.array(b), np.array(x0)
    )
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

    iteration_callback = TrajectoryIterationCallback(f)

    conjugate_direction_method_for_quadratic(
        np.array(Q),
        np.array(b),
        np.array(x0),
        iteration_callback=iteration_callback,
    )

    first_x, first_fx = iteration_callback.trajectory[0]
    last_x, last_fx = iteration_callback.trajectory[-1]

    assert approx_equal(first_x, [0, 0])
    assert approx_equal(first_fx, 0)
    assert approx_equal(last_x, [-2.5, 1])
    assert approx_equal(last_fx, -1.25)
    assert len(iteration_callback.trajectory) == 2


def test_that_number_of_iterations_is_not_greater_than_number_of_dimensions():
    # f(x, z) = 100(z - x)^2 + (1 - x)^2 = 101x^2 - 2x - 200xz + 100z^2 + 1
    f = lambda x: 100 * (x[1] - x[0]) ** 2 + (1 - x[0]) ** 2
    Q = np.array(
        [
            [202, -200],
            [-200, 200],
        ]
    )
    b = np.array([-2, 0])
    x0 = np.array([-1, 1])

    iteration_callback = TrajectoryIterationCallback(f)

    res = conjugate_direction_method_for_quadratic(
        Q, b, x0, iteration_callback=iteration_callback
    )

    assert len(iteration_callback.trajectory) <= 3
    assert approx_equal(res, [1, 1])

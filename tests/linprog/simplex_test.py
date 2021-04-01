import numpy as np

import methopt.linprog.simplex as simplex


def test_system1():
    c = np.array([0, 2, -1, 1], dtype="float64")
    A = np.array(
        [
            [2, 1, 1, 0],
            [1, 2, 0, 1],
        ],
        dtype="float64",
    )
    b = np.array([6, 6], dtype="float64")
    x = simplex.solve_canonical(c, A, b)
    assert np.all(x == [2, 2, 0, 0])


def test_system2():
    c = -np.array([-6, -1, -4, +5])
    A = np.array(
        [
            [3, 1, -1, 1],
            [5, 1, 1, -1],
        ],
        dtype="float64",
    )
    b = np.array([4, 4], dtype="float64")
    x = simplex.solve_canonical(c, A, b)
    assert np.all(x == [0, 4, 0, 0])


def test_system3():
    c = -np.array([-1, -2, -3, 1], dtype="float64")
    A = np.array(
        [
            [1, -3, -1, -2],
            [1, -1, 1, 0],
        ],
        dtype="float64",
    )
    b = np.array([-4, 0], dtype="float64")
    x = simplex.solve_canonical(c, A, b)
    assert np.all(x == [2, 2, 0, 0])


def test_system4():
    c = -np.array([-1, -2, -1, 3, -1], dtype="float64")
    A = np.array(
        [
            [1, 1, 0, 2, 1],
            [1, 1, 1, 3, 2],
            [0, 1, 1, 2, 1],
        ],
        dtype="float64",
    )
    b = np.array([5, 9, 6], dtype="float64")
    x = simplex.solve_canonical(c, A, b)
    assert np.all(x == [3, 2, 4, 0, 0])


def test_system5():
    c = -np.array([-1, -1, -1, 1, -1])
    A = np.array(
        [[1, 1, 2, 0, 0], [0, -2, -2, 1, -1], [1, -1, 6, 1, 1]],
        dtype="float64",
    )
    b = np.array([4, -6, 12], dtype="float64")
    x = simplex.solve_canonical(c, A, b)
    assert np.allclose(x, [4, 0, 0, 1, 7])


def test_system6():
    c = -np.array([-1, 4, -3, 10])
    A = np.array(
        [[1, 1, -1, -10], [1, 14, 10, -10]],
        dtype="float64",
    )
    b = np.array([0, 11], dtype="float64")
    x = simplex.solve_canonical(c, A, b)
    print(x)
    assert np.allclose(x, [1, 0, 1, 0])


def test_system7():
    c = -np.array([-1, 5, 1, -1, 0, 0])  # + y1, y2
    A = np.array(
        [[1, 3, 3, 1, 1, 0], [2, 0, 3, -1, 0, 1]],
        dtype="float64",
    )
    b = np.array([3, 4], dtype="float64")
    x = simplex.solve_canonical(c, A, b)
    assert np.allclose(x[:4], [7 / 3, 0, 0, 2 / 3])


def test_system8():
    c = -np.array([-1, -1, 1, -1, 2])
    A = np.array(
        [[3, 1, 1, 1, -2], [6, 1, 2, 3, -4], [10, 1, 3, 6, -7]],
        dtype="float64",
    )
    b = np.array([10, 20, 30], dtype="float64")
    x = simplex.solve_canonical(c, A, b)
    assert np.allclose(x, [10, 0, 0, 0, 10])

import numpy as np

from methopt.conjugate_direction_method import conjugate_direction_method

EPS = 1e-7


def approx_equal(a, b, eps=EPS):
    return np.all(abs(a - b) < eps)


def test1():
    Q = [[2]]
    b = [0]
    x0 = [-6]

    res = conjugate_direction_method(np.array(Q), np.array(b), np.array(x0))
    print(res)
    assert approx_equal(res, [0])


def test2():
    Q = [[2]]
    b = [20]
    x0 = [0]

    res = conjugate_direction_method(np.array(Q), np.array(b), np.array(x0))
    print(res)
    assert approx_equal(res, [-10])


def test2():
    Q = [[20]]
    b = [-100]
    x0 = [0]

    res = conjugate_direction_method(np.array(Q), np.array(b), np.array(x0))
    print(res)
    assert approx_equal(res, [5])


def test3():
    Q = [
        [2, 0],
        [0, 2]
    ]
    b = [0, 0]
    x0 = [10, 10]

    res = conjugate_direction_method(np.array(Q), np.array(b), np.array(x0))
    print(res)
    assert approx_equal(res, [0, 0])


def test4():
    Q = [
        [2, 0],
        [0, 2]
    ]
    b = [1, 0]
    x0 = [10, 10]

    res = conjugate_direction_method(np.array(Q), np.array(b), np.array(x0))
    print(res)
    assert approx_equal(res, [-0.5, 0])


def test5():
    Q = [
        [4, 0],
        [0, 2]
    ]
    b = [1, 0]
    x0 = [10, 10]

    res = conjugate_direction_method(np.array(Q), np.array(b), np.array(x0))
    print(res)
    assert approx_equal(res, [-0.25, 0])


def test6():
    Q = [[2, 4],
         [4, 10]]
    b = [1, 0]
    x0 = [10, 10]

    res = conjugate_direction_method(np.array(Q), np.array(b), np.array(x0))
    print(res)
    assert approx_equal(res, [-2.5, 1])

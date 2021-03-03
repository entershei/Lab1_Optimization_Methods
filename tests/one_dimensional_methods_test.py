from src.methopt.one_dimensional_methods import (
    dichotomy,
    golden_section,
    fibonacci_method,
    pre_calc_for_fibonacci_method,
)

EPSILONS = [0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
MAX_STEPS = 100


def approx_equal(a, b, eps):
    return abs(a - b) <= eps


def internal_test_function(
    f, x_ans, a0, b0, method_name, max_steps=MAX_STEPS, epsilons=None
):
    if epsilons is None:
        epsilons = EPSILONS
    for eps in epsilons:
        if method_name == "dichotomy":
            x, _, _ = dichotomy(eps, a0, b0, f, max_steps)
        elif method_name == "golden_section":
            x, _, _ = golden_section(eps, a0, b0, f, max_steps)
        else:
            x, _, _ = fibonacci_method(
                eps, a0, b0, f, max_steps, pre_calc_for_fibonacci_method(max_steps)
            )
        assert approx_equal(x, x_ans, eps)


def test_parabole1():
    f = lambda x: (x - 3) ** 2 + 8
    x_ans = 3
    a0 = -500
    b0 = 500
    internal_test_function(f, x_ans, a0, b0, "dichotomy")
    internal_test_function(f, x_ans, a0, b0, "golden_section")
    internal_test_function(f, x_ans, a0, b0, "fibonacci")


def test_parabole2():
    f = lambda x: (x + 5.5) ** 2
    x_ans = -5.5
    a0 = -500
    b0 = 500
    max_steps = 1000
    internal_test_function(f, x_ans, a0, b0, "dichotomy", max_steps)
    internal_test_function(f, x_ans, a0, b0, "golden_section", max_steps)
    internal_test_function(f, x_ans, a0, b0, "fibonacci", max_steps)


def test_parabole3():
    f = lambda x: (x + 5.5) ** 2 - 100
    x_ans = 0
    a0 = 0
    b0 = 1000
    internal_test_function(f, x_ans, a0, b0, "dichotomy")
    internal_test_function(f, x_ans, a0, b0, "golden_section")
    internal_test_function(f, x_ans, a0, b0, "fibonacci")


def test_line():
    f = lambda x: -x
    x_ans = 100
    a0 = 0
    b0 = 100
    internal_test_function(f, x_ans, a0, b0, "dichotomy")
    internal_test_function(f, x_ans, a0, b0, "golden_section")
    internal_test_function(f, x_ans, a0, b0, "fibonacci")


def test_sqrt():
    f = lambda x: (x + 1) ** 0.5
    x_ans = -1
    a0 = -1
    b0 = 500
    internal_test_function(f, x_ans, a0, b0, "dichotomy")
    internal_test_function(f, x_ans, a0, b0, "golden_section")
    internal_test_function(f, x_ans, a0, b0, "fibonacci")


def test_hard_f1():
    f = lambda x: (x - 1) / (1 - x ** 3)
    x_ans = -0.5
    a0 = -500
    b0 = 500
    internal_test_function(f, x_ans, a0, b0, "dichotomy")
    internal_test_function(f, x_ans, a0, b0, "golden_section")
    internal_test_function(f, x_ans, a0, b0, "fibonacci")


def test_hard_f2():
    f = lambda x: (x - 1) / (1 - x ** 3)
    x_ans = -0.5
    a0 = -500
    b0 = 5000
    internal_test_function(f, x_ans, a0, b0, "dichotomy")
    internal_test_function(f, x_ans, a0, b0, "golden_section")
    internal_test_function(f, x_ans, a0, b0, "fibonacci")


def test_hard_f3():
    f = lambda x: (x - 1) / (1 - x ** 3)
    x_ans = -0.5
    a0 = -1
    b0 = 1
    internal_test_function(f, x_ans, a0, b0, "dichotomy")
    internal_test_function(f, x_ans, a0, b0, "golden_section")
    internal_test_function(f, x_ans, a0, b0, "fibonacci")

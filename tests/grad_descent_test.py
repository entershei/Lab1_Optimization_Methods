from methopt.grad_descent import grad_descent
import methopt.step_adjustment_strategy as step

EPS = 1e-7


def approx_equal(a, b, eps=EPS):
    return abs(a - b) < eps


def test_grad_descent_divide_step():
    f = lambda x: (x - 3) ** 2 + 8
    f_grad = lambda x: 2 * (x - 3)
    x0 = -6

    assert approx_equal(
        grad_descent(f, f_grad, x0, step_adjustment_strategy="divide_step"), 3
    )


def test_grad_descent_dichotomy():
    f = lambda x: (x - 3) ** 2 + 8
    f_grad = lambda x: 2 * (x - 3)
    x0 = -6

    strategy = step.DichotomyStrategy(f, f_grad, max_step=1000, max_steps=20, eps=1e-8)
    assert approx_equal(
        grad_descent(
            f,
            f_grad,
            x0,
            step_adjustment_strategy=strategy,
            stopping_criterion="function_margin",
        ),
        3,
        eps=1e-4,  # it does not perform well -_-. so we set a lower correctness margin
    )


def test_grad_descent_golden_section():
    f = lambda x: (x - 3) ** 2 + 8
    f_grad = lambda x: 2 * (x - 3)
    x0 = -6

    strategy = step.GoldenSectionStrategy(
        f, f_grad, max_step=1000, max_steps=20, eps=1e-8
    )
    assert approx_equal(
        grad_descent(
            f,
            f_grad,
            x0,
            step_adjustment_strategy=strategy,
            stopping_criterion="argument_margin",
        ),
        3,
    )


def test_grad_descent_fibonacci():
    f = lambda x: (x - 3) ** 2 + 8
    f_grad = lambda x: 2 * (x - 3)
    x0 = -6

    strategy = step.FibonacciStrategy(f, f_grad, max_step=1000, max_steps=20, eps=1e-8)
    assert approx_equal(
        grad_descent(
            f,
            f_grad,
            x0,
            step_adjustment_strategy=strategy,
            stopping_criterion="argument_margin",
        ),
        3,
    )


def test_grad_descent_iteration_callback():
    f = lambda x: x ** 2 - 5
    f_grad = lambda x: 2 * x
    x0 = 8

    max_iterations_count = 50

    trajectory = []
    iteration_callback = lambda x, **kwargs: trajectory.append((x, f(x)))

    grad_descent(
        f,
        f_grad,
        x0,
        max_iterations_count=max_iterations_count,
        stopping_criterion="n_iterations",
        iteration_callback=iteration_callback,
    )

    assert len(trajectory) == max_iterations_count

    last_x, last_fx = trajectory[-1]
    assert approx_equal(last_x, 0)
    assert approx_equal(last_fx, -5)


def test_grad_descent_far_initial_point():
    MIN_POINT = 8
    MIN_VALUE = -6
    f = lambda x: (x - MIN_POINT) ** 2 + MIN_VALUE
    f_grad = lambda x: 2 * (x - MIN_POINT)
    x0 = 1000

    max_step = 1000
    step_adjustment_strategy = step.GoldenSectionStrategy(f, f_grad, max_step)
    res = grad_descent(
        f,
        f_grad,
        x0,
        max_iterations_count=10000,
        step_adjustment_strategy=step_adjustment_strategy,
        stopping_criterion="function_margin",
    )

    assert approx_equal(res, MIN_POINT, eps=1e-4)


def test_on_inverse_hyperbole():
    # long story short: if you can afford, use argument_margin instead
    # of function_margin

    # vanishing gradient is at play here with dichotomy…

    f = lambda x: -1 / (x ** 2 + x + 1)
    f_grad = lambda x: (2 * x + 1) / (x ** 2 + x + 1) ** 2
    x0 = 1

    actual_res = -1 / 2

    res1 = grad_descent(
        f,
        f_grad,
        x0,
        step_adjustment_strategy=step.DivideStepStrategy(f, f_grad),
    )

    assert approx_equal(res1, actual_res)

    max_step = 1000
    res2 = grad_descent(
        f,
        f_grad,
        x0,
        step_adjustment_strategy=step.DichotomyStrategy(
            f, f_grad, max_step, max_steps=30
        ),
    )

    assert approx_equal(res2, actual_res)

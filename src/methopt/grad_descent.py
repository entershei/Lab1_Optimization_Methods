import methopt.step_adjustment_strategy as strat


def grad_descent(
    f,
    f_grad,
    x0,
    max_iterations_count=1000,
    step_adjustment_strategy="divide_step",
    initial_step=1,
    eps=None,
):
    """Find an approximation of a local minimum of the function.

    Arguments:
    'f' — a diffentiable, (locally) convex function to find minumum of
    'f_grad' — a gradient of 'f'
    'x0' — initial guess, does not need to be correct

    Keyword arguments:

    'max_iterations_count' — a maximum number of iterations to do
    before stopping at the best guess.

    'step_adjustment_strategy' — a name of a strategy or an object of
    strategy that adjusts a gradient step on every
    iteration. Available strategies can be found in
    ./step_adjustment_strategy.py

    'intial_step' — an initial gradient step.

    'eps' — a margin for floating-point comparisons, default is 1e-7

    """

    if eps is None:
        eps = 1e-7

    if step_adjustment_strategy == "divide_step":
        step_adjustment_strategy = strat.DivideStepStrategy(f, f_grad, eps=eps)
    elif not isinstance(step_adjustment_strategy, strat.StepAdjustmentStrategy):
        raise GradDescentException(
            f"Unknown step adjustment strategy: {step_adjustment_strategy}"
        )

    x = x0
    step_prev = initial_step
    for iteration_no in range(max_iterations_count):
        step = step_adjustment_strategy(x, step_prev, iteration_no)
        x_new = x - step * f_grad(x)

        # TODO add a function parameter to choose a stopping criterion
        if abs(x - x_new) < eps:
            break

        step_prev = step
        x = x_new

    return x


class GradDescentException(Exception):
    pass

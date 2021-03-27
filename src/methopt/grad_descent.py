import numpy as np

import methopt.step_adjustment_strategy as strat


def grad_descent(
    f,
    f_grad,
    x0,
    max_iterations_count=1000,
    step_adjustment_strategy="divide_step",
    initial_step=1,
    eps=None,
    stopping_criterion=None,
    iteration_callback=None,
    **kwargs,
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

    'stopping_criterion' — a criterion for when to stop
    searching. Possible values: "argument_margin" — stop when a
    distance between consecutive points is smaller than 'eps',
    "function_margin" — stop when a distance between consecutive
    function values is smaller than 'eps', "n_iterations" — stop after
    'max_iterations_count' iterations. Default is "argument_margin".

    'iteration_callback' — a function from (x, iteration_no) where x
    is a point in the search space at the current iteration
    #iteration_no. Please note that the function is going to be called
    with named arguments, so name them as specified here. Default is
    no-op.

    Example: `iteration_callback=lambda x, **kwargs: print(x, f(x))`.

    Note that we use **kwargs to ignore arguments that we don't need.

    """

    if eps is None:
        eps = 1e-7

    if step_adjustment_strategy == "divide_step":
        step_adjustment_strategy = strat.DivideStepStrategy(f, f_grad, eps=eps)
    elif not isinstance(step_adjustment_strategy, strat.StepAdjustmentStrategy):
        raise GradDescentException(
            f"Unknown step adjustment strategy: {step_adjustment_strategy}"
        )

    if stopping_criterion is None or stopping_criterion == "argument_margin":
        is_finished = lambda x, x_new: np.linalg.norm(x - x_new) < eps
    elif stopping_criterion == "function_margin":
        is_finished = lambda x, x_new: abs(f(x) - f(x_new)) < eps
    elif stopping_criterion == "n_iterations":
        # we do "max_iterations_count" anyway
        is_finished = lambda x, x_new: False
    else:
        raise GradDescentException(f"Unknown stopping_criterion: {stopping_criterion}")

    if iteration_callback is None:
        iteration_callback = lambda **kwargs: ()  # no-op

    x = x0
    step_prev = initial_step
    for iteration_no in range(max_iterations_count):
        iteration_callback(x=x, iteration_no=iteration_no)
        step = step_adjustment_strategy(x, step_prev, iteration_no)
        assert step >= 0
        x_new = x - step * f_grad(x)

        if is_finished(x, x_new):
            break

        step_prev = step
        x = x_new

    return x


class GradDescentException(Exception):
    pass

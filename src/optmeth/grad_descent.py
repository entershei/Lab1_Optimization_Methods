EPS = 1e-7


def divide_step_strategy(f, x, step_prev, iteration_no, mk_new_x_callback):
    """A strategy to find a new gradient step.

    This works by taking the old step and dividing it by two, until a
    function value in the next point is not less than the value in the
    current point.

    Once in 100 iterations we try to make a gradient step much bigger,
    so as too speed up a search.

    """
    if iteration_no % 100 == 0:  # try bigger step
        step = step_prev * 256
    else:
        step = step_prev

    x_new = mk_new_x_callback(step)
    fx = f(x)
    # HMM perhaps need to make floating-point-tailored comparison
    while f(x_new) >= fx and step > EPS:
        step /= 2
        x_new = mk_new_x_callback(step)

    return step


def grad_descent(
    f,
    f_grad,
    x0,
    max_iterations_count=1000,
    step_adjustment_strategy="divide_step",
    initial_step=1,
):
    """Find an approximation of a local minimum of the function.

    Arguments:
    'f' — a diffentiable, (locally) convex function to find minumum of
    'f_grad' — a gradient of 'f'
    'x0' — initial guess, does not need to be correct

    Keyword arguments:

    'max_iterations_count' — a maximum number of iterations to do
    before stopping at the best guess.

    'step_adjustment_strategy' — a name of a strategy that adjusts a
    gradient step on every iteration. Available strategies:
    '\"grad_descent.divide_step_strategy\"'

    'intial_step' — an initial gradient step.

    """

    if step_adjustment_strategy == "divide_step":
        step_adjustment_strategy = divide_step_strategy
    else:
        raise GradDescentException(
            f"Unknown step adjustment strategy: {step_adjustment_strategy}"
        )

    x = x0
    step_prev = initial_step
    mk_new_x_callback = lambda step: x - step * f_grad(x)
    for iteration_no in range(max_iterations_count):
        step = step_adjustment_strategy(
            f,
            x,
            step_prev=step_prev,
            iteration_no=iteration_no,
            mk_new_x_callback=mk_new_x_callback,
        )
        x_new = mk_new_x_callback(step)

        # TODO add a function parameter to choose a stopping criterion
        if abs(x - x_new) < EPS:
            break

        step_prev = step
        x = x_new

    return x


class GradDescentException(Exception):
    pass

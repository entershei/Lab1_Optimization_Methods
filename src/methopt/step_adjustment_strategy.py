import methopt.one_dimensional_methods as impl


class StepAdjustmentStrategy:
    """A base class for strategies to find a new gradient step"""

    def __init__(self, f, f_grad, eps=None):
        """Arguments:

        'f' — a diffentiable, (locally) convex function to find
        minumum of

        'f_grad' — a gradient of 'f'

        Keyword arguments:

        'eps' — a margin for floating-point comparisons, default is 1e-7

        """
        if eps is None:
            eps = 1e-7

        self.f = f
        self.f_grad = f_grad
        self.eps = eps

    def __call__(self, x, step_prev, iteration_no=None):
        raise NotImplemented


class DivideStepStrategy(StepAdjustmentStrategy):
    """A strategy to find a new gradient step by by taking the old step
    and dividing it by two, until a # function value in the next point is
    not less than the value in the # current point.

    Once in 100 iterations we try to make a gradient step much bigger,
    so as too speed up a search.

    """

    def __call__(self, x, step_prev, iteration_no):
        if iteration_no % 100 == 0:  # try bigger step
            step = step_prev * 256
        else:
            step = step_prev

        x_new = x - step * self.f_grad(x)
        fx = self.f(x)
        while self.f(x_new) >= fx and step > self.eps:
            step /= 2
            x_new = x - step * self.f_grad(x)

        return step


class OneDimOptimizationStrategy(StepAdjustmentStrategy):
    """A common class for all adjustment strategies that search for an
    optimal step via one-dimensional optimization methods

    """

    def __init__(self, f, f_grad, opt_method, max_step, max_steps=None, eps=None):
        super().__init__(f, f_grad, eps)

        if max_steps is None:
            max_steps = 10

        self.max_step = max_step
        self.max_steps = max_steps
        self.opt_method = opt_method

    def __call__(self, x, step_prev=None, iteration_no=None):
        g = lambda step: self.f(x - step * self.f_grad(x))
        step, *_ = self.opt_method(self.eps, 0, self.max_step, g, self.max_steps)
        return step


class DichotomyStrategy(OneDimOptimizationStrategy):
    def __init__(self, f, f_grad, max_step, max_steps=None, eps=None):
        super().__init__(
            f,
            f_grad,
            opt_method=impl.dichotomy,
            max_step=max_step,
            max_steps=max_steps,
            eps=eps,
        )


class GoldenSectionStrategy(OneDimOptimizationStrategy):
    def __init__(self, f, f_grad, max_step, max_steps=None, eps=None):
        super().__init__(
            f,
            f_grad,
            opt_method=impl.golden_section,
            max_step=max_step,
            max_steps=max_steps,
            eps=eps,
        )


class FibonacciStrategy(OneDimOptimizationStrategy):
    def __init__(self, f, f_grad, max_step, max_steps=None, eps=None):
        super().__init__(
            f,
            f_grad,
            opt_method=None,  # we are going to set it later
            max_step=max_step,
            max_steps=max_steps,
            eps=eps,
        )
        fibo = [0 for _ in range(self.max_steps + 1)]  # memoize fibonacci numbers
        fibo[0] = 1
        fibo[1] = 1
        impl.compute_fibonacci(max_steps, fibo)
        self.opt_method = lambda *args, **kwargs: impl.fibonacci_method(
            *args, **kwargs, fibonacci=fibo
        )

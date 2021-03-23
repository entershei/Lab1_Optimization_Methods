import numpy as np

from methopt.conjugate_direction_method import \
    conjugate_direction_method_for_quadratic


class DivideStepStrategy:
    def __init__(self, f, eps=None):
        if eps is None:
            eps = 1e-7

        self.f = f
        self.eps = eps

    def __call__(self, x, x_wave, step_prev, iteration_no):
        if iteration_no % 100 == 0:
            step = step_prev * 256
        else:
            step = step_prev

        x_new = x + step * x_wave
        fx = self.f(x)
        while self.f(x_new) >= fx and step > self.eps:
            step /= 2
            x_new = x + step * x_wave

        return step


def newtons_method(
        f,
        f_H,
        f_grad,
        x0,
        initial_step=1,
        eps=1e-7,
        max_iterations_count=1000,
        iteration_callback=None,
):
    def call_on_H(H, x):
        res = np.zeros(H.shape)
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                res[i][j] = H[i][j](x)
        return res

    def call_on_grad(grad, x):
        res = np.zeros(grad.shape)
        for i in range(grad.shape[0]):
            res[i] = grad[i](x)
        return res

    if iteration_callback is None:
        iteration_callback = lambda **kwargs: ()

    step_adjustment_strategy = DivideStepStrategy(f, eps)
    step_prev = initial_step
    f_prev = f(x0)
    x_prev = x0
    iteration_callback(x=x0, iteration_no=0)
    for k in range(1, max_iterations_count):
        # psi(x) = (H(x_prev)(x - x_prev),x - x_prev)
        #        + (grad(x_prev), x - x_prev) + f(x_prev)
        # f(x_prev) doesn't affect min's coordinates
        # min(psi(x)) = x_wave + x_prev
        x_wave = conjugate_direction_method_for_quadratic(
            call_on_H(f_H, x_prev), call_on_grad(f_grad, x_prev), x0
        )
        step = step_adjustment_strategy(x_prev, x_wave, step_prev, k)
        # xk = x_prev + step((x_wave + x_prev) - x_prev)
        xk = x_prev + step * x_wave
        iteration_callback(x=xk, iteration_no=k)
        fk = f(xk)
        if np.all(abs(f_prev - fk) < eps):
            return xk
        x_prev = xk
        f_prev = fk
        step_prev = step
    return x_prev

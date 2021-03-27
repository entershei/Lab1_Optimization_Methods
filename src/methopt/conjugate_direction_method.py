import numpy as np

from methopt.grad_descent import grad_descent
from methopt.step_adjustment_strategy import DivideStepStrategy


def conjugate_direction_method_for_quadratic(
    Q,
    b,
    x0,
    max_iterations_count=1000,
    iteration_callback=None,
    eps=None,
):
    # f(x) = 0.5 (Qx, x) + (b, x)
    if iteration_callback is None:
        iteration_callback = lambda **kwargs: ()

    if eps is None:
        eps = 1e-8

    if np.all(b == 0):
        iteration_callback(x=b, iteration_no=0)
        return b

    iteration_callback(x=x0, iteration_no=0)
    w1 = -Q @ x0 - b
    p1 = w1
    if np.linalg.norm(p1) == 0:
        return x0
    h1 = np.dot(p1, p1) / np.dot(Q @ p1, p1)
    x1 = x0 + h1 * p1

    x_prev = x1
    w_prev = w1
    p_prev = p1
    h_prev = h1
    for k in range(1, max_iterations_count):
        wk = w_prev - h_prev * (Q @ p_prev)
        yk = np.dot(Q @ p_prev, wk) / np.dot(Q @ p_prev, p_prev)
        pk = wk - yk * p_prev
        if abs(np.linalg.norm(pk)) < eps:
            return x_prev
        hk = np.dot(wk, pk) / np.dot(Q @ pk, pk)
        xk = x_prev + hk * pk
        iteration_callback(x=xk, iteration_no=k)

        p_prev = pk
        w_prev = wk
        x_prev = xk
        h_prev = hk
    return x_prev


def conjugate_direction_method(
    f,
    f_grad,
    x0,
    max_iterations_count=1000,
    iteration_callback=None,
    eps=1e-3,  # Search accuracy
    **kwargs,
):
    if iteration_callback is None:
        iteration_callback = lambda **kwargs: ()

    iteration_callback(x=x0, iteration_no=0)
    w1 = -f_grad(x0)
    p1 = w1
    if np.linalg.norm(w1) < eps:
        return x0
    # x1 = x0 + h1 * p1

    x_prev = x0
    w_prev = w1
    p_prev = p1
    for k in range(1, max_iterations_count):
        psi = lambda chi: f(x_prev + chi * p_prev)
        grad_psi = lambda chi: np.dot(f_grad(x_prev + chi * p_prev), p_prev)
        hk = grad_descent(
            psi,
            grad_psi,
            0,
            initial_step=1e-3,
            eps=1e-7,
            step_adjustment_strategy=DivideStepStrategy(psi, grad_psi, eps=1e-11),
        )
        xk = x_prev + hk * p_prev
        iteration_callback(x=xk, iteration_no=k)
        wk = -f_grad(xk)

        if abs(np.linalg.norm(wk)) < eps:
            return xk

        yk = max(0, np.dot(wk - w_prev, wk) / np.dot(wk, wk))
        pk = wk + yk * p_prev
        p_prev = pk
        w_prev = wk
        x_prev = xk
    return x_prev

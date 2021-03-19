import numpy as np


def conjugate_direction_method(
    Q,
    b,
    x0,
    max_iterations_count=1000,  # todo: I think it isn't needed
    iteration_callback=None,
):
    # f(x) = 0.5 (Qx, x) + (b, x)
    if iteration_callback is None:
        iteration_callback = lambda **kwargs: ()

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
        if np.linalg.norm(pk) == 0:
            return x_prev
        hk = np.dot(wk, pk) / np.dot(Q @ pk, pk)
        xk = x_prev + hk * pk
        iteration_callback(x=xk, iteration_no=k)

        p_prev = pk
        w_prev = wk
        x_prev = xk
        h_prev = hk
    return x_prev

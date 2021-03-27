import numpy as np
import math
from methopt.conjugate_direction_method import (
    conjugate_direction_method_for_quadratic,
    conjugate_direction_method,
)
from methopt.newtons_method import newtons_method
import csv

EPS = 1e-7


def approx_equal(a, b, eps=EPS):
    return np.all(abs(a - b) < eps)


def print_result(f_out, results):
    f_out = "method_comparison/min_by_cdm_and_newtons/" + f_out + ".csv"
    with open(f_out, "w", newline="") as csv_file:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for log in results:
            writer.writerow(log)


def get_f_res(f_res, is_max):
    return f_res if not is_max else -1 * f_res


def find_min_by_different_methods(
    f,
    f_grad,
    H,
    x0s,
    x_min,
    f_min,
    f_out,
    eps_cdm=EPS,
    eps_newtons=EPS,
    run_cdm=True,
    run_newtons=True,
    Q=None,
    b=None,
    is_max=False,
):
    def iteration_callback(x, iteration_no):
        nonlocal iterations
        iterations = max(iterations, iteration_no)

    to_log = []
    for x0 in x0s:
        x0 = np.array(x0, dtype=np.float64)

        if Q and run_cdm:
            iterations = 0
            res_cdm_q = conjugate_direction_method_for_quadratic(
                np.array(Q), np.array(b), x0, iteration_callback=iteration_callback
            )
            f_res = get_f_res(f(res_cdm_q), is_max)
            if x_min is not None:
                assert approx_equal(res_cdm_q, x_min)
                assert approx_equal(f_res, f_min)
            to_log.append(
                {
                    "method": "cdm_q",
                    "x_0": x0,
                    "x_min": res_cdm_q,
                    "f(x_min)": f_res,
                    "iterations": iterations,
                }
            )

        if run_cdm:
            iterations = 0
            res_cdm = conjugate_direction_method(
                f, f_grad, x0, iteration_callback=iteration_callback
            )
            f_res = get_f_res(f(res_cdm), is_max)
            if x_min is not None:
                assert approx_equal(res_cdm, x_min, eps_cdm)
                assert approx_equal(f_res, f_min, eps_cdm)
            to_log.append(
                {
                    "method": "cdm",
                    "x_0": x0,
                    "x_min": res_cdm,
                    "f(x_min)": f_res,
                    "iterations": iterations,
                }
            )

        if run_newtons:
            iterations = 0
            res_newtons = newtons_method(
                f, H, f_grad, x0, iteration_callback=iteration_callback
            )
            f_res = get_f_res(f(res_newtons), is_max)
            if x_min is not None:
                assert approx_equal(res_newtons, x_min, eps_newtons)
                assert approx_equal(f_res, f_min, eps_newtons)
            to_log.append(
                {
                    "method": "newtons",
                    "x_0": x0,
                    "x_min": res_newtons,
                    "f(x_min)": f_res,
                    "iterations": iterations,
                }
            )

    print_result(f_out, to_log)


def quadratic_func(run_cdm=True, run_newtons=True):
    # f(x, z) = 100(z - x)^2 + (1 - x)^2 = 101x^2 - 2x - 200xz + 100z^2 + 1
    f = lambda x: 100 * (x[1] - x[0]) ** 2 + (1 - x[0]) ** 2
    f_grad = lambda x: np.array(
        [202 * x[0] - 2 - 200 * x[1], -200 * x[0] + 200 * x[1]], dtype="float64"
    )
    Q = [
        [202, -200],
        [-200, 200],
    ]
    H = lambda x: np.array(Q, dtype="float64")
    b = [-2, 0]
    x0s = [[1, 1], [0, 0], [-1, -1], [1.87, -2.3], [1, -1], [10, 10], [-13.2, 4.5]]
    find_min_by_different_methods(
        f,
        f_grad,
        H,
        x0s,
        x_min=[1, 1],
        f_min=0,
        f_out="quadratic_function",
        eps_cdm=1e-3,
        eps_newtons=1e-7,
        run_cdm=run_cdm,
        run_newtons=run_newtons,
        Q=Q,
        b=b,
    )


def rosenbrock_func(run_cdm, run_newtons):
    # f(x, z) = 100(z - x^2)^2 + (1 - x)^2 = 100x^4 + x^2 - 200x^2z - 2x + 100z^2 + 1
    f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    f_grad = lambda x: np.array(
        [
            400 * x[0] ** 3 + 2 * x[0] - 400 * x[0] * x[1] - 2,
            -200 * x[0] ** 2 + 200 * x[1],
        ],
        dtype="float64",
    )
    H = lambda x: np.array(
        [
            [1200 * x[0] ** 2 + 2 - 400 * x[1], -400 * x[0]],
            [-400 * x[0], 200],
        ],
        dtype="float64",
    )
    x0s = [[1, 1], [0, 0], [-1, -1], [1, -1], [1.87, -2.3], [1.2, 1.67]]
    find_min_by_different_methods(
        f,
        f_grad,
        H,
        x0s,
        x_min=[1, 1],
        f_min=0,
        f_out="rosenbrock_function",
        eps_cdm=1e-2,
        eps_newtons=1e-3,
        run_cdm=run_cdm,
        run_newtons=run_newtons,
    )


def test_func(run_cdm, run_newtons):
    f = lambda x: -2 * math.exp(
        -(((x[0] - 1) / 2) ** 2) - (x[1] - 1) ** 2
    ) - 3 * math.exp(-(((x[0] - 2) / 3) ** 2) - ((x[1] - 3) / 2) ** 2)
    f_grad = lambda x: np.array(
        [
            2
            * (x[0] - 2)
            / 3
            * math.exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
            + (x[0] - 1) * math.exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
            3
            / 2
            * (x[1] - 3)
            * math.exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
            + 4 * (x[1] - 1) * math.exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
        ],
        dtype="float64",
    )
    H = lambda x: np.array(
        [
            [
                (-4 / 27 * (x[0] - 2) ** 2 + 2 / 3)
                * math.exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
                + (1 - (x[0] - 1) ** 2 / 2)
                * math.exp(-((x[0] - 1) ** 2) / 4 - (x[1] - 1) ** 2),
                -1
                / 3
                * (x[0] - 2)
                * (x[1] - 3)
                * math.exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
                - 2
                * (x[0] - 1)
                * (x[1] - 1)
                * math.exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
            ],
            [
                -1
                / 3
                * (x[0] - 2)
                * (x[1] - 3)
                * math.exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
                - 2
                * (x[0] - 1)
                * (x[1] - 1)
                * math.exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
                (-3 / 4 * (x[1] - 3) ** 2 + 3 / 2)
                * math.exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
                + (4 - 8 * (x[1] - 1) ** 2)
                * math.exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
            ],
        ],
        dtype="float64",
    )
    x0s = [[3, 3], [0, 0], [-1, -1], [1.5, 1.35], [3.5, 3.5], [1.2, 1.2], [0.3, 0.5]]
    find_min_by_different_methods(
        f,
        f_grad,
        H,
        x0s,
        x_min=None,
        f_min=None,
        f_out="test_function",
        run_cdm=run_cdm,
        run_newtons=run_newtons,
        is_max=True,
    )


def main():
    quadratic_func(run_cdm=True, run_newtons=True)
    rosenbrock_func(run_cdm=True, run_newtons=True)
    test_func(run_cdm=True, run_newtons=True)


if __name__ == "__main__":
    main()

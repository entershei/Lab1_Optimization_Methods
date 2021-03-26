from memory_profiler import memory_usage
from timeit import timeit
import numpy as np
import sys
import csv
from methopt.conjugate_direction_method import conjugate_direction_method
from methopt.grad_descent import grad_descent
from methopt.newtons_method import newtons_method

ITERATION_CNT = 3

methods = {
    "newtons_method": newtons_method,
    "grad_descent": grad_descent,
    "conjugate_direction_method": conjugate_direction_method,
}


def time_cnt(**kwargs):
    passed_kwargs = {name: kwargs[name] for name in ["f", "f_H", "f_grad", "x0"]}
    if "method" in kwargs:
        method = kwargs["method"]
        return round(
            timeit(
                f"{method.__name__}(f=f, f_H=f_H, f_grad=f_grad, x0=x0)",
                setup=f"from methopt.{method.__name__} import {method.__name__}",
                number=ITERATION_CNT,
                globals=passed_kwargs,
            )
            / ITERATION_CNT,
            3,
        )
    else:
        print(
            "newtons_method",
            round(
                timeit(
                    "newtons_method(f=f, f_H=f_H, f_grad=f_grad, x0=x0)",
                    setup="from methopt.newtons_method import newtons_method",
                    number=ITERATION_CNT,
                    globals=passed_kwargs,
                )
                / ITERATION_CNT,
                3,
            ),
            "s",
        )
        print(
            "conjugate directions",
            round(
                timeit(
                    "conjugate_direction_method(f=f, f_H=f_H, f_grad=f_grad, x0=x0)",
                    setup="from methopt.conjugate_direction_method import conjugate_direction_method",
                    number=ITERATION_CNT,
                    globals=passed_kwargs,
                )
                / ITERATION_CNT,
                3,
            ),
            "s",
        )
        print(
            "grad_descent",
            round(
                timeit(
                    "grad_descent(f=f, f_H=f_H, f_grad=f_grad, x0=x0)",
                    setup="from methopt.grad_descent import grad_descent",
                    number=ITERATION_CNT,
                    globals=passed_kwargs,
                )
                / ITERATION_CNT,
                3,
            ),
            "s",
        )


# @profile needed for manual launch and memory profiling
def mem_cnt(**kwargs):
    method = kwargs["method"]
    passed_kwargs = {name: kwargs[name] for name in ["f", "f_H", "f_grad", "x0"]}
    return memory_usage((method, (), passed_kwargs), interval=0.001, max_usage=True)


def iter_cnt(**kwargs):
    counter = 0

    def inc(**kwargs):
        nonlocal counter
        counter += 1

    iteration_callback = inc

    method = kwargs["method"]
    passed_kwargs = {name: kwargs[name] for name in ["f", "f_H", "f_grad", "x0"]}
    passed_kwargs["iteration_callback"] = iteration_callback
    res = method(**passed_kwargs)
    # print(res) ## check for convergence
    return counter


d = {
    "formula": ["x^2 + 10y^2 + 5", "4x^6 + 10z^2 - 4xz + 10z"],
    "f": [
        lambda x: x[0] ** 2 + 10 * x[1] ** 2 + 5,
        lambda x: 4 * x[0] ** 6 + 10 * x[1] ** 2 - 4 * x[0] * x[1] + 10 * x[1],
    ],
    "f_grad": [
        lambda x: np.array([2 * x[0], 20 * x[1]]),
        lambda x: np.array(
            [
                24 * x[0] ** 5 - 4 * x[1],
                20 * x[1] - 4 * x[0] + 10,
            ],
            dtype="float64",
        ),
    ],
    "f_H": [
        lambda x: np.array(
            [
                [2, 0],
                [0, 20],
            ]
        ),
        lambda x: np.array(
            [
                [120 * x[0] ** 4, -4],
                [-4, 20],
            ],
            dtype="float64",
        ),
    ],
    "x0": [np.array([5, -7]), np.array([3, 3], dtype="float64")],
}

if __name__ == "__main__":
    if len(sys.argv) == 2:
        method_name = sys.argv[1]
        params = {key: d[key][1] for key in ["formula", "f", "f_H", "f_grad", "x0"]}
        params["method"] = methods[method_name]
        mem_cnt(**params)
    else:
        for formula, f, f_grad, f_H, x0 in zip(
            d["formula"], d["f"], d["f_grad"], d["f_H"], d["x0"]
        ):
            with open(
                f"method_comparison/1-2/{formula}.csv", "w+", newline=""
            ) as table:
                writer = csv.DictWriter(table, fieldnames=["method", "time", "iter"])
                writer.writeheader()
                for method_name in methods:
                    cur_time = time_cnt(
                        method=methods[method_name],
                        formula=formula,
                        f=f,
                        f_grad=f_grad,
                        f_H=f_H,
                        x0=x0,
                    )
                    cur_cnt = iter_cnt(
                        method=methods[method_name], f=f, f_grad=f_grad, f_H=f_H, x0=x0
                    )
                    writer.writerow(
                        {"method": method_name, "time": cur_time, "iter": cur_cnt}
                    )

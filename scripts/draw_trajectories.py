import os
import itertools
import uuid

import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, sin, cos

from methopt.conjugate_direction_method import conjugate_direction_method
from methopt.grad_descent import grad_descent
from methopt.newtons_method import newtons_method

from methopt.utils import TrajectoryIterationCallback

graphics_dir = "images/comparison"

methods = [
    grad_descent,
    conjugate_direction_method,
    newtons_method,
]

LIMIT = 8


def in_limits(point) -> bool:
    return np.all(-LIMIT < point) and np.all(point < LIMIT)


def escape_filename(filename: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_X500, filename))


def draw_trajectory_with_method(method, marker, **kwargs):
    iteration_callback = TrajectoryIterationCallback(kwargs["f"])
    method(iteration_callback=iteration_callback, max_iterations_count=30, **kwargs)
    points, _ = zip(*iteration_callback.trajectory)
    xs, ys = zip(*points)
    plt.plot(xs, ys, label=method.__name__, marker=marker)
    return points


def markers_cycle():
    yield from itertools.cycle(range(4, 12))


def draw_2d(formula: str, f, f_grad, f_H, x0, comment: str = None):
    min_x, max_x = LIMIT, -LIMIT
    min_y, max_y = LIMIT, -LIMIT
    for method, marker in zip(methods, markers_cycle()):
        points = draw_trajectory_with_method(
            method, marker, f=f, f_grad=f_grad, f_H=f_H, x0=x0, eps=1e-3
        )
        points = filter(in_limits, points)
        xs, ys = zip(*points)
        min_x, max_x = min(min_x, *xs), max(max_x, *xs)
        min_y, max_y = min(min_y, *ys), max(max_y, *ys)

    dx, dy = max_x - min_x, max_y - min_y
    expansion = 0.1
    x = np.arange(min_x - dx * expansion - 1, max_x + dx * expansion + 1, 0.01)
    y = np.arange(min_y - dy * expansion - 1, max_y + dy * expansion + 1, 0.01)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = f((xx, yy))

    title = f"${formula}$, x0 = {x0}"
    if comment is not None:
        title = f"{comment}: {title}"

    plt.legend()
    plt.contour(x, y, z)
    plt.suptitle(title, fontsize=16, wrap=True, usetex=True)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.savefig(f"images/comparison/{escape_filename(title)}.jpg")
    plt.close()


def draw_f1():
    f = lambda x: 2 * x[0] ** 2 + x[1] ** 2
    f_grad = lambda x: np.array([4 * x[0], 2 * x[1]])
    f_H = lambda x: np.array(
        [
            [4, 0],
            [0, 2],
        ]
    )
    x0 = np.array([1, 10])
    draw_2d("2x^2+y^2+5", f, f_grad, f_H, x0)


def draw_f2():
    f = lambda x: x[0] ** 2 + 10 * x[1] ** 2 + 5
    f_grad = lambda x: np.array([2 * x[0], 20 * x[1]])
    f_H = lambda x: np.array(
        [
            [2, 0],
            [0, 20],
        ]
    )
    x0 = np.array([5, -7])
    draw_2d("x^2+10y^2+5", f, f_grad, f_H, x0)


def draw_f3():
    # f = 4x^6 + 10z^2 - 4xz + 10z
    f = lambda x: 4 * x[0] ** 6 + 10 * x[1] ** 2 - 4 * x[0] * x[1] + 10 * x[1]
    f_grad = lambda x: np.array(
        [
            24 * x[0] ** 5 - 4 * x[1],
            20 * x[1] - 4 * x[0] + 10,
        ],
        dtype="float64",
    )
    f_H = lambda x: np.array(
        [
            [120 * x[0] ** 4, -4],
            [-4, 20],
        ],
        dtype="float64",
    )
    x0 = np.array([3, 3], dtype="float64")
    draw_2d("4x^6 + 10z^2 - 4xz + 10z", f, f_grad, f_H, x0)


def draw_f4():
    # f = (x^2 + y^2)^2 + y^3
    f = lambda x: (x[0] ** 2 + x[1] ** 2) ** 2 + x[1] ** 3
    f_grad = lambda x: np.array(
        [
            2 * (x[0] ** 2 + x[1] ** 2) * 2 * x[0],
            2 * (x[0] ** 2 + x[1] ** 2) * 2 * x[1] + 3 * x[1] ** 2,
        ]
    )
    f_H = lambda x: np.array(
        [
            [12 * x[0] ** 2 + 4 * x[1] ** 2, 8 * x[1] * x[0]],
            [8 * x[1] * x[0], 4 * x[0] ** 2 + 12 * x[1] ** 2 + 6 * x[1]],
        ]
    )
    x0 = np.array([-1.5, -1.5])
    draw_2d("(x^2 + y^2)^2 + y^3", f, f_grad, f_H, x0)


def dont_draw_f1():
    # f = e^{-y^2} * sin(x)
    f = lambda x: exp(-x[1] ** 2) * sin(x[0])
    f_grad = lambda x: np.array(
        [exp(-x[1] ** 2) * cos(x[0]), -2 * exp(-x[1] ** 2) * x[1] * sin(x[0])]
    )
    f_H = lambda x: np.array(
        [
            [-exp(-x[1] ** 2) * sin(x[0]), -2 * exp(-x[1] ** 2) * x[1] * cos(x[0])],
            [
                -2 * exp(-x[1] ** 2) * x[1] * cos(x[0]),
                4 * exp(-x[1] ** 2) * x[1] ** 2 * sin(x[0])
                - 2 * exp(-x[1] ** 2) * sin(x[0]),
            ],
        ]
    )
    x0 = np.array([-0.4, 2])
    draw_2d("e^{-y^2} * sin(x)", f, f_grad, f_H, x0)


def draw_f5():
    # f = 100(y - x)^2 + (1 - x)^2 = 101x^2 - 2x - 200xz + 100z^2 + 1
    f = lambda x: 100 * (x[1] - x[0]) ** 2 + (1 - x[0]) ** 2
    f_grad = lambda x: np.array(
        [202 * x[0] - 2 - 200 * x[1], -200 * x[0] + 200 * x[1]], dtype="float64"
    )
    f_H = lambda x: np.array(
        [
            [202, -200],
            [-200, 200],
        ]
    )
    x0 = np.array([1.87, -2.3])
    draw_2d("100(y - x)^2 + (1 - x)^2", f, f_grad, f_H, x0)


def draw_f6():
    # f = 100(y - x^2)^2 + (1 - x)^2 (rosenbrok)
    f = lambda x: 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    f_grad = lambda x: np.array(
        [
            400 * x[0] ** 3 + 2 * x[0] - 400 * x[0] * x[1] - 2,
            -200 * x[0] ** 2 + 200 * x[1],
        ],
        dtype="float64",
    )
    f_H = lambda x: np.array(
        [
            [1200 * x[0] ** 2 + 2 - 400 * x[1], -400 * x[0]],
            [-400 * x[0], 200],
        ],
        dtype="float64",
    )
    x0 = np.array([1.2, -1.5])
    draw_2d("100(y - x^2)^2 + (1 - x)^2", f, f_grad, f_H, x0, comment="rosenbrok")


def draw_f7():
    # f = 2 * e^{-((x - 1) / 2)^2 - (y - 1)^2} + 3 * e^{-((x - 2) / 3)^2 - ((y - 3) / 2)^2}
    f = lambda x: -2 * exp(-(((x[0] - 1) / 2) ** 2) - (x[1] - 1) ** 2) - 3 * exp(
        -(((x[0] - 2) / 3) ** 2) - ((x[1] - 3) / 2) ** 2
    )
    f_grad = lambda x: np.array(
        [
            2 * (x[0] - 2) / 3 * exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
            + (x[0] - 1) * exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
            3 / 2 * (x[1] - 3) * exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
            + 4 * (x[1] - 1) * exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
        ],
        dtype="float64",
    )
    f_H = lambda x: np.array(
        [
            [
                (-4 / 27 * (x[0] - 2) ** 2 + 2 / 3)
                * exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
                + (1 - (x[0] - 1) ** 2 / 2)
                * exp(-((x[0] - 1) ** 2) / 4 - (x[1] - 1) ** 2),
                -1
                / 3
                * (x[0] - 2)
                * (x[1] - 3)
                * exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
                - 2
                * (x[0] - 1)
                * (x[1] - 1)
                * exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
            ],
            [
                -1
                / 3
                * (x[0] - 2)
                * (x[1] - 3)
                * exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
                - 2
                * (x[0] - 1)
                * (x[1] - 1)
                * exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
                (-3 / 4 * (x[1] - 3) ** 2 + 3 / 2)
                * exp(-1 / 9 * (x[0] - 2) ** 2 - 1 / 4 * (x[1] - 3) ** 2)
                + (4 - 8 * (x[1] - 1) ** 2)
                * exp(-1 / 4 * (x[0] - 1) ** 2 - (x[1] - 1) ** 2),
            ],
        ],
        dtype="float64",
    )
    x0 = np.array([1.5, -0.2], dtype="float64")
    draw_2d(
        "2 * e^{-((x - 1) / 2)^2 - (y - 1)^2} + 3 * e^{-((x - 2) / 3)^2 - ((y - 3) / 2)^2}",
        f,
        f_grad,
        f_H,
        x0,
    )


def main():
    os.makedirs(graphics_dir, exist_ok=True)
    draw_f1()
    draw_f2()
    draw_f3()
    draw_f4()
    draw_f5()
    draw_f6()
    draw_f7()


if __name__ == "__main__":
    main()

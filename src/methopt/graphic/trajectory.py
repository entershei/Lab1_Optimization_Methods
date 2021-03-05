from methopt.grad_descent import grad_descent
import numpy as np
import matplotlib.pyplot as plt


def draw_2d(f, f_grad, x0, title, step_adjustment_strategy=None):
    if step_adjustment_strategy is None:
        step_adjustment_strategy = "divide_step"

    trajectory = []
    iteration_callback = lambda x, **kwargs: trajectory.append((x, f(x)))
    grad_descent(
        f,
        f_grad,
        x0,
        max_iterations_count=20,
        step_adjustment_strategy=step_adjustment_strategy,
        stopping_criterion="n_iterations",
        iteration_callback=iteration_callback,
    )
    points, values = zip(*trajectory)
    xs, ys = zip(*points)

    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    dx, dy = max_x - min_x, max_y - min_y
    expansion = 1
    x = np.arange(min_x - dx * expansion - 1, max_x + dx * expansion + 1, 0.01)
    y = np.arange(min_y - dy * expansion - 1, max_y + dy * expansion + 1, 0.01)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = f((xx, yy))

    plt.contour(x, y, z)
    plt.scatter(xs, ys, s=10, c="black", edgecolors="black")
    plt.plot(xs, ys, c="black")
    plt.suptitle(title, fontsize=16)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.savefig(f"images/{title}.jpg")
    plt.close()


def main():
    f0 = lambda x: 2 * x[0] ** 2 + x[1] ** 2
    f0_grad = lambda x: np.array([4 * x[0], 2 * x[1]])
    draw_2d(f0, f0_grad, np.array([1, 10]), f"2x^2+y^2+5_(1;10)")

    f1 = lambda x: x[0] ** 2 + 10 * x[1] ** 2 + 5
    f1_grad = lambda x: np.array([2 * x[0], 20 * x[1]])
    draw_2d(f1, f1_grad, np.array([5, -7]), f"x^2+10y^2+5_(5;-7)")

    f3 = lambda x: (x[0] - 3) ** 2 + (x[1] - 1) ** 2 + 1
    f3_grad = lambda x: np.array([2 * x[0] - 6, 2 * x[1] - 2])
    draw_2d(f3, f3_grad, np.array([-3, 3]), title="(x-3)^2+(y-1)^2+1_(-3;3)")


if __name__ == "__main__":
    main()

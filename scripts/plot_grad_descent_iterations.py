import matplotlib.pyplot as plt

import methopt.step_adjustment_strategy as strategy
from methopt.grad_descent import grad_descent


def count_iterations_for_strategy(f, f_grad, x0, step_adjustment_strategy):
    iterations_count = 0

    def iteration_callback(iteration_no, **kwargs):
        nonlocal iterations_count
        iterations_count = max(iterations_count, iteration_no)

    stopping_criterion = "function_margin"
    max_iterations_count = 1_000_000

    res = grad_descent(
        f,
        f_grad,
        x0,
        max_iterations_count=max_iterations_count,
        step_adjustment_strategy=step_adjustment_strategy,
        stopping_criterion=stopping_criterion,
        iteration_callback=iteration_callback,
        eps=1e-4,
    )

    return iterations_count


def bars_for_function(function_label, f, f_grad, x0):
    MAX_STEP = 1000
    params = [
        (strategy.DivideStepStrategy(f, f_grad), "divide step"),
        (strategy.DichotomyStrategy(f, f_grad, MAX_STEP), "dichotomy"),
        (strategy.GoldenSectionStrategy(f, f_grad, MAX_STEP), "golden section"),
        (strategy.FibonacciStrategy(f, f_grad, MAX_STEP), "fibonacci"),
    ]
    strategies, labels = zip(*params)
    results = [
        count_iterations_for_strategy(f, f_grad, x0, strategy)
        for strategy in strategies
    ]

    print(results)

    fig = plt.figure(num=function_label)
    ax = fig.add_subplot()
    ax.set_title(function_label)
    print(f"{labels=}, {results=}")
    ax.bar(labels, results)
    ax.yaxis.get_major_locator().set_params(integer=True)

    return fig


def main():
    def f1(x):
        return (x - 5) ** 2 + 18

    def f1_grad(x):
        return 2 * (x - 5)

    def f2(x):
        return -1 / (x ** 2 + x + 1)

    def f2_grad(x):
        return (2 * x + 1) / (x ** 2 + x + 1) ** 2

    functions = [
        ("hyperbole", f1, f1_grad, 1231),
        ("inverse-hyperbole", f2, f2_grad, 1),
    ]

    labels, *_ = zip(*functions)

    figs = [bars_for_function(*function) for function in functions]

    for fig, label in zip(figs, labels):
        fig.savefig(f"method_comparison/grad_descent/{label}")
    plt.show()


if __name__ == "__main__":
    main()

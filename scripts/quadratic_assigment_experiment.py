import csv
import numpy as np
from random import uniform

from methopt.grad_descent import grad_descent
import methopt.step_adjustment_strategy as step
from methopt.quadratic_assignment import (
    generate_quadratic_assignment
)


# Run experiment and write results in CSV format. Headers: name,n,k,iters.
def run_experiment(strategy_name):
    with open(
            "method_comparison/quadratic_assignment/" + strategy_name + ".csv",
            "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["name", "n", "k", "iters"])
        writer.writeheader()
        for n in range(2, 12):
            for k in range(1, 11):
                iterations_count = 0

                def iteration_callback(iteration_no, **kwargs):
                    nonlocal iterations_count
                    iterations_count = max(iterations_count, iteration_no)

                x0 = np.array([uniform(-10, 10) for i in range(n)])
                f, grad = generate_quadratic_assignment(n, k)
                f_grad = lambda x: np.array(
                    [gr(x) for gr in grad])  # map(lambda gr: gr(x), grad)

                name_to_strategy = {
                    'divide_step': 'divide_step',
                    'dichotomy': step.DichotomyStrategy(f, f_grad,
                                                        max_step=1000,
                                                        max_steps=20, eps=1e-8),
                    'golden_section': step.GoldenSectionStrategy(f, f_grad,
                                                                 max_step=1000,
                                                                 max_steps=20,
                                                                 eps=1e-8),
                    'fibonacci': step.FibonacciStrategy(f, f_grad,
                                                        max_step=1000,
                                                        max_steps=20, eps=1e-8)
                }
                if strategy_name not in name_to_strategy:
                    print("Invalid strategy" + strategy_name)
                    return

                g = grad_descent(f, f_grad, x0,
                                 step_adjustment_strategy=name_to_strategy[
                                     strategy_name],
                                 iteration_callback=iteration_callback)
                writer.writerow(
                    {"name": strategy_name, "n": str(n), "k": str(k),
                     "iters": str(iterations_count)})


def main():
    run_experiment("divide_step")
    run_experiment("golden_section")
    run_experiment("dichotomy")
    run_experiment("fibonacci")


if __name__ == "__main__":
    main()

import csv

from src.methopt.one_dimensional_methods import (
    pre_calc_for_fibonacci_method,
    dichotomy,
    golden_section,
    fibonacci_method,
)

EPSILONS = [0.5, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]


def approx_equal(a, b, eps):
    return abs(a - b) <= eps


def print_matrix(f_out, results, description):
    f_out = "method_comparison/one_dim/" + f_out + ".csv"
    with open(f_out, "w", newline="") as csvfile:
        fieldnames = ["eps"] + list(next(iter(results.values())).keys()) + [description]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for eps in results.keys():
            results[eps].update({"eps": eps})
            writer.writerow(results[eps])


def compare_methods(f, x_ans, a0, b0, max_steps, f_name, description):
    fibonacci = pre_calc_for_fibonacci_method(max_steps)
    results = {}

    for eps in EPSILONS:
        x_d, iterations_d, calls_d = dichotomy(eps, a0, b0, f, max_steps)
        x_g, iterations_g, calls_g = golden_section(eps, a0, b0, f, max_steps)
        x_f, iterations_f, calls_f = fibonacci_method(
            eps, a0, b0, f, max_steps, fibonacci
        )
        assert approx_equal(x_d, x_ans, eps)
        assert approx_equal(x_g, x_ans, eps)
        assert approx_equal(x_f, x_ans, eps)

        results[eps] = {
            "dichotomy_iterations": iterations_d,
            "dichotomy_function_calls": calls_d,
            "golden_section_iterations": iterations_g,
            "golden_section_function_calls": calls_g,
            "fibonacci_iterations": iterations_f,
            "fibonacci_function_calls": calls_f,
        }

    print_matrix(f_name, results, description)


def main():
    max_steps = 1000

    f = lambda x: (x - 3) ** 2 + 8
    x_ans = 3
    a0 = -500
    b0 = 500
    compare_methods(
        f,
        x_ans,
        a0,
        b0,
        max_steps,
        "parabole",
        "f(x) = (x - 3)^2 + 8, a = " + str(a0) + ", b = " + str(b0),
    )

    f = lambda x: -x
    x_ans = 100
    a0 = 0
    b0 = 100
    compare_methods(
        f,
        x_ans,
        a0,
        b0,
        max_steps,
        "line",
        "f(x) = -x, a = " + str(a0) + ", b = " + str(b0),
    )

    f = lambda x: (x + 1) ** 0.5
    x_ans = -1
    a0 = -1
    b0 = 500
    compare_methods(
        f,
        x_ans,
        a0,
        b0,
        max_steps,
        "sqrt",
        "f(x) = sqrt(x + 1), a = " + str(a0) + ", b = " + str(b0),
    )

    f = lambda x: (x - 1) / (1 - x ** 3)
    x_ans = -0.5
    a0 = -500
    b0 = 500
    compare_methods(
        f,
        x_ans,
        a0,
        b0,
        max_steps,
        "hard_function1",
        "f(x) = (x - 1) / (1 - x^3), a = " + str(a0) + ", b = " + str(b0),
    )

    f = lambda x: (x - 1) / (1 - x ** 3)
    x_ans = -0.5
    a0 = -1
    b0 = 1
    compare_methods(
        f,
        x_ans,
        a0,
        b0,
        max_steps,
        "hard_function2",
        "f(x) = (x - 1) / (1 - x^3), a = " + str(a0) + ", b = " + str(b0),
    )


if __name__ == "__main__":
    main()

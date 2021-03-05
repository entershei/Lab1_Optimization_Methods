PHI = (1 + 5 ** 0.5) / 2
FIBONACCI_MAX_STEPS = 52

# For each method:
# f: R -> R - continuous and unimodal on the start segment [a0, b0]


# On each iteration |b - a| ~ |b_previous - a_previous| / 2
# Iterations will be ~ log2(|b0 - a0| / eps)
# Each iteration calls f twice
# Returns x*, number of iterations, number of calls f
def dichotomy(eps, a, b, f, max_steps):
    sigma = 1e-8
    iterations = 0
    while abs(b - a) > 2 * eps and iterations < max_steps:
        x = (a + b) / 2
        x1 = x - sigma
        x2 = x + sigma
        f1 = f(x1)
        f2 = f(x2)
        if f1 < f2:
            b = x2
        else:
            a = x1
        iterations += 1

    x = (a + b) / 2
    function_calls = iterations * 2
    return x, iterations, function_calls


# |b1 - a1| = |b0 - a0| / PHI
# |b_k - a_k| = |b_(k - 2) - a_(k - 2)| - |b_(k - 1) - a_(k - 1)|
# Iterations will be ~ 1 / log_0.7(2 * |b0 - a0| / eps)
# Each iteration calls f once
# Returns x*, number of iterations, number of calls f
def golden_section(eps, a, b, f, max_steps):
    x1 = b - (b - a) / PHI
    x2 = a + (b - a) / PHI
    iterations = 0
    know_f1 = True
    f_old = f(x1)
    while abs(b - a) > 2 * eps and iterations < max_steps:
        iterations += 1
        if know_f1:
            f1 = f_old
            f2 = f(x2)
        else:
            f1 = f(x1)
            f2 = f_old

        if f1 < f2:
            b = x2
            x2 = x1
            x1 = b - (b - a) / PHI
            know_f1 = False
            f_old = f1
        elif f1 > f2:
            a = x1
            x1 = x2
            x2 = a + (b - a) / PHI
            know_f1 = True
            f_old = f2
        else:
            break

    x = (a + b) / 2
    return x, iterations, iterations + 1


def find_n(len_ab, eps, fibonacci, max_steps):
    left = 0
    right = max_steps
    while right - left > 1:
        m = (left + right) // 2
        if 2 * len_ab < eps * fibonacci[m]:
            right = m
        else:
            left = m
    return right


# |b_k - a_k| ~ |b_0 - a_0| * Fib(n - k - 1) / Fib(n - k)
# Each iteration calls f once
# Iterations will be = n, where n: |b_0 - a_0| / Fib(n) < eps / 2
# Returns x*, number of iterations, number of calls f
def fibonacci_method(eps, a, b, f, max_steps, fibonacci):
    if max_steps > FIBONACCI_MAX_STEPS:
        max_steps = FIBONACCI_MAX_STEPS
    n = find_n(abs(b - a), eps, fibonacci, max_steps)

    iterations = 1
    x1 = a + fibonacci[n - 2] / fibonacci[n] * abs(b - a)
    x2 = a + fibonacci[n - 1] / fibonacci[n] * abs(b - a)
    know_f1 = True
    f_old = f(x1)
    while iterations < n - 2:
        if know_f1:
            f1 = f_old
            f2 = f(x2)
        else:
            f1 = f(x1)
            f2 = f_old
        if f1 < f2:
            b = x2
            x2 = x1
            know_f1 = False
            f_old = f1
            x1 = a + fibonacci[n - iterations - 2] / fibonacci[n - iterations] * (b - a)
        else:
            a = x1
            x1 = x2
            know_f1 = True
            f_old = f2
            x2 = a + fibonacci[n - iterations - 1] / fibonacci[n - iterations] * (b - a)
        iterations += 1

    sigma = 1e-8
    x2 = x1 + sigma
    if f(x1) < f(x2):
        b = x2

    x = (a + b) / 2
    return x, iterations, iterations + 1


def compute_fibonacci(n, fibonacci):
    if fibonacci[n] != 0:
        return fibonacci[n]
    fibonacci[n] = compute_fibonacci(n - 1, fibonacci) + compute_fibonacci(
        n - 2, fibonacci
    )
    return fibonacci[n]


def pre_calc_for_fibonacci_method(max_steps):
    if max_steps > FIBONACCI_MAX_STEPS:
        max_steps = FIBONACCI_MAX_STEPS
    fibonacci = [0] * (max_steps + 1)
    fibonacci[0] = 1
    fibonacci[1] = 1
    compute_fibonacci(max_steps, fibonacci)
    return fibonacci

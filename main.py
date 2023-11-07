import numpy as np


def main():
    print(eval_poly1(np.array([1, 2, 3], float), 2))
    print(diff_poly1(np.array([1, 2, 3], float), 2, True))
    print(diff_poly1(np.array([1, 2, 3], float), 2, False))
    print(coeffs_poly3(2))
    a = np.array([14, 4, 1, -6, 0, 1, -2, 0, 0, 1], float)
    x0 = np.array([0, 0, 0], float)
    x = np.array([1, 3, -2], float)
    print(eval_poly3(2, a, x0))
    print(eval_poly3(2, a, x))
    print(diff_poly3(2, a, x0))
    print(diff_poly3(2, a, x))
    print(find_minimum(2, a, x0))
    return


def eval_poly1(a: np.ndarray, x: float) -> float:
    f = 0
    for i in range(len(a)):
        f += a[i] * x**i
    return f


def diff_poly1(a: np.ndarray, x: float, mode: bool) -> float:
    if mode:
        df = 0
        for i in range(1, len(a)):
            df += a[i] * i * x**(i-1)
    else:
        epsilon = 1e-6
        df = (eval_poly1(a, x + epsilon) - eval_poly1(a, x)) / epsilon
    return df


def coeffs_poly3(d: int) -> int:
    n = 0
    for i in range(d+1):
        for j in range(d+1):
            for k in range(d+1):
                if i+j+k <= d:
                    n += 1
    return n


def eval_poly3(d: int, a: np.ndarray, x: np.ndarray) -> float:
    f = 0
    n = 0
    for i in range(d+1):
        for j in range(d+1):
            for k in range(d+1):
                if i+j+k <= d:
                    f += a[n] + x[0]**i * x[1]**j * x[2]**k
                    n += 1
    return f


def diff_poly3(d: int, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    epsilon = 1e-6
    df = np.zeros(3,)
    f = eval_poly3(d, a, x)
    for i in range(3):
        x[i] += epsilon
        df[i] = (eval_poly3(d, a, x) - f) / epsilon
        x[i] -= epsilon
    return df


def find_minimum(d: int, a: np.ndarray, x: np.ndarray) -> (np.ndarray, int):
    max_iter = 1000
    delta = 1e-4
    eta = 0.1
    # eta = 0.5
    # eta = 1.0
    for i in range(max_iter):
        df = diff_poly3(d,a,x)
        x -= eta * df
        if np.linalg.norm(df) < delta:
            break
    return x, i


main()

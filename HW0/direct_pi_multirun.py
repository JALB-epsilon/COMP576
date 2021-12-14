import random
import sys
import matplotlib.pyplot as plt

def direct_pi(N):
    n_hits = 0
    for i in range(N):
        x, y = random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)
        if x ** 2 + y ** 2 < 1.0:
            n_hits += 1
    return 4.0 * n_hits / float(n_trial)

def MonteCarloApproxMult(n_trial, n_runs):
    result = []
    for run in range(n_runs):
        result.append(direct_pi(n_trial))
    return result


if __name__ == "__main__":
    input = sys.stdin.read()
    n_trial, n_runs = map(int, input.split())
    X = MonteCarloApproxMult(n_trial, n_runs)
    plt.hist(X, bins=30)
    plt.title('Pi Distribution')
    plt.show()

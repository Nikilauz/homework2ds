import numpy as np
import matplotlib.pyplot as plt

def autodot(x):
    return np.inner(x, x)

def guess_grad_lip(x_samples, grad_vals, old_guess=0.):
    lip = old_guess
    size = len(x_samples)
    for _ in range(100 * size):
        [i, j] = np.random.randint(0, size, 2)
        if np.all(x_samples[i] == x_samples[j]):
            continue
        lip = max(lip, np.linalg.norm(grad_vals[i]-grad_vals[j], 2) / np.linalg.norm(x_samples[i]-x_samples[j], 2))
    return lip

def beta_from_equation_3(x, x_old, f_x, f_x_old, df_x_old):
    return 2 * (f_x - f_x_old - np.dot(df_x_old, x - x_old)) / autodot(x - x_old)


def argmin_F(f_oracle, df_oracle, g_oracle, prox_transform, hyperpars):
    RAND_SIZE = hyperpars["RAND_SIZE"]
    N = hyperpars["N"]
    GUESS_EFFORT = hyperpars["GUESS_EFFORT"]
    TOLERANCE = hyperpars["TOLERANCE"]
    STEPOUT = hyperpars["STEPOUT"]

    xs = [np.random.uniform(-RAND_SIZE, RAND_SIZE, N) for _ in range(GUESS_EFFORT)]
    dxs = [np.linalg.norm(xs[i] - xs[i-1], 2) for i in range(1, len(xs))]
    fs = [f_oracle(x) for x in xs]
    dfs = [df_oracle(x) for x in xs]
    Fs = [fs[i] + g_oracle(x) for (i, x) in enumerate(xs)]
    betas = [guess_grad_lip(xs, dfs)]
    # print(xs)
    # print(fs)
    # print(dfs)
    # print(Fs)
    print("initial beta: " + str(betas[0]))

    solved = lambda : dxs[-1] < TOLERANCE
    recompute_beta = lambda : len(xs) > STEPOUT / 4 and (dxs[-1] > 1e3 or dxs[-1] - dxs[-2] < TOLERANCE)

    for i in range(STEPOUT):
        x_new = prox_transform(betas[-1], xs[-1], dfs[-1])
        dx_new = np.linalg.norm(xs[-1] - xs[-2], 2)
        beta_new = beta_from_equation_3(xs[-1], xs[-2], fs[-1], fs[-2], dfs[-2]) if recompute_beta() else betas[-1]

        print("dx=" + str(dx_new) + " (iteration " + str(i) + "/" + str(STEPOUT) + ")...", end="\r")

        xs += [x_new]
        dxs += [dx_new]
        fs += [f_oracle(x_new)]
        dfs += [df_oracle(x_new)]
        Fs += [fs[-1] + g_oracle(x_new)]
        betas += [beta_new]

        if solved():
            print("\nconverged after " + str(i) + " steps")
            break
    else:
        print("\ndid not converge")

    i_star = np.argmin(Fs)
    x_star = xs[i_star]

    print("best index: " + str(i_star - GUESS_EFFORT))

    plt.plot(Fs)
    #plt.plot(betas)
    #plt.plot([np.linalg.norm(xs[i]-xs[i-1], 2) for i in range(1, len(xs))])

    return x_star
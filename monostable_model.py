import numpy as np
import matplotlib.pyplot as plt
from random import gauss, choice
from scipy.stats import levy_stable


def random_force_gauss(D, h, **kwargs):
    return (h * D) ** 0.5 * gauss(0, 1)  # losowa siła

def model_force_square(x, a, b):
    return -a*(x - b)

def random_force_levy(D, h, alpha=2, beta=0., loc=0., scale=1.):
    print(alpha)
    r = levy_stable.rvs(alpha=alpha, beta=beta, loc=loc, scale=scale)
    return (h * D) ** 0.5 * r  # losowa siła

def ion_channel_model(a=1, closed=(-1, 5000), opened=(1, 2500), D=0.5, delta_t=0.01, records=50000, model_force=model_force_square, random_force=random_force_gauss, **force_params):
    t = 0
    # losujemy czy kanał początkowo jest otwarty czy zamknięty
    b = choice([closed, opened])
    x = np.array([model_force(0, a, b[0]) * delta_t + random_force(D, delta_t, force_params)], dtype=np.float32)
    times = np.array([t], dtype=np.float32)
    # losujemy tau, czyli jak długo kanał jest O/Z
    tau = np.random.exponential(b[1])
    while t < records:
        x = np.append(
        x, (
            x[t - 1] +
            model_force(x[t - 1], a, b[0]) * delta_t +
            random_force(D, delta_t)
            )
    )
        times = np.append(times, t * delta_t)
        t += 1
        tau -= 1
        if(tau <= 0):
            if(b[0] == closed[0]):
                b=opened
            else:
                b=closed
            tau = np.random.exponential(b[1])

    plt.clf()
    plt.plot(times, x)
    plt.show()
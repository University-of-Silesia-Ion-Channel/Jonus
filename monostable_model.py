import numpy as np
import matplotlib.pyplot as plt
from random import gauss, choice

def random_force_gauss(D, h):
    return (h * D) ** 0.5 * gauss(0, 1)  # losowa siła

def model_force_square(x, a, b):
    return -a*(x - b)

def monostable_model(a, D=1.0, delta_t=0.001, records=1000, model_force=model_force_square, random_force=random_force_gauss):


    t = 0
    # losujemy czy kanał początkowo jest otwarty czy zamknięty
    b = choice([-1, 1])
    x = np.array([model_force(0, a, b) * delta_t + random_force(D, delta_t)], dtype=np.float32)
    times = np.array([t], dtype=np.float32)
    # losujemy tau, czyli jak długo kanał jest O/Z
    tau = np.random.exponential(10000)
    while t < records:
    # print(self.x[t - 1] + self.F(self.x[t - 1], a, b) * delta_t)
        x = np.append(
        x, (
            x[t - 1] +
            model_force(x[t - 1], a, b) * delta_t +
            random_force(D, delta_t)
            )
    )
        times = np.append(times, t * delta_t)
        t += 1
        tau -= 1
        if(tau <= 0):
            b = -b
            tau = np.random.exponential(10000)

    plt.plot(times, x)
    plt.show()
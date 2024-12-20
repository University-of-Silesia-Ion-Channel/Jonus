import numpy as np
import matplotlib.pyplot as plt
from random import gauss, choice

def random_force_gauss(D, h):
    return (h * D) ** 0.5 * gauss(0, 1)  # losowa siła

def model_force_square(x, a, b):
    return -a*(x - b)

def ion_channel_model(a=1, closed=(-1, 5000), opened=(1, 2500), D=0.5, delta_t=0.01, records=50000, model_force=model_force_square, random_force=random_force_gauss):
    t = 0
    # losujemy czy kanał początkowo jest otwarty czy zamknięty
    b = choice([closed, opened])
    x = np.array([model_force(0, a, b[0]) * delta_t + random_force(D, delta_t)], dtype=np.float32)
    times = np.array([t], dtype=np.float32)
    # losujemy tau, czyli jak długo kanał jest O/Z
    tau = np.random.exponential(b[1])
    while t < records:
    # print(self.x[t - 1] + self.F(self.x[t - 1], a, b) * delta_t)
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

    plt.plot(times, x)
    plt.show()
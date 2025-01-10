import numpy as np
import matplotlib.pyplot as plt
from random import gauss, choice
from scipy.stats import levy_stable


def random_force_gauss(D, h, records, args):
    """Function generates random force values using gaussian distribution.

    Args:
        D (float): Needs to be similar to h in terms of order of magnitude.
        h (float): Needs to be similar to D in terms of order of magnitude.
        records (int): Number of records to generate.
        args (dict): Placeholder for additional arguments.

    Returns:
        np.array: Array with random force values.
    """
    return (h * D) ** 0.5 * gauss(0, 1, size=records)  # losowa siła

def model_force_square(x, a, b):
    """Function calculates force value using square function.

    Args:
        x (float): Position value.
        a (float): Coefficient.
        b (float): Coefficient.

    Returns:
        float: Force value at x.
    """
    return -a*(x - b)

# takes two common parameters and dictionary with 4 parameters needed for levy_stable.rvs function
def random_force_levy(D, h, records, args):
    """Function generates random force values using levy_stable distribution.

    Args:
        D (float): Needs to be similar to h in terms of order of magnitude.
        h (float): Needs to be similar to D in terms of order of magnitude.
        records (int): Number of records to generate.
        args (dict): Placeholder for additional arguments. Contains: alpha, beta, location, scale.
        

    Returns:
        np.array: Array with random force values.
    """
    r = levy_stable.rvs(alpha=args['alpha'], beta=args['beta'], loc=args['location'], scale=args['scale'], size=records)
    return (h * D) ** 0.5 * r  # losowa siła

def ion_channel_model(a=1, closed=(-1, 5000), opened=(1, 2500), D=0.5, delta_t=0.01, records=50000, model_force=model_force_square, random_force=random_force_gauss, **force_params):
    """Function, that generates a plot of time series of ion channel model.

    Args:
        a (int, optional): One of the coefficients of force. Defaults to 1.
        closed (tuple, optional): Tuple where first item is state of closed ion channel and second is value to be passed to np.random.exponential to generate time the channel is closed. Defaults to (-1, 5000).
        opened (tuple, optional): Tuple where first item is state of closed ion channel and second is value to be passed to np.random.exponential to generate time the channel is opened. Defaults to (1, 2500).
        D (float, optional): Needs to be similar to delta_t in terms of order of magnitude. Defaults to 0.5.
        delta_t (float, optional): Needs to be similar to D in terms of order of magnitude. Defaults to 0.01.
        records (int, optional): Number of records to generate.. Defaults to 50000.
        model_force (func, optional): Parameter to pass model force. Defaults to model_force_square.
        random_force (func, optional): Parameter to pass random force that generates noise. Defaults to random_force_gauss.
        **force_params: Additional parameters for random force function.
    """
    
    t = 0
    # losujemy czy kanał początkowo jest otwarty czy zamknięty
    b = choice([closed, opened])
    random_force_values = random_force(D, delta_t, records, force_params)
    x = np.array([model_force(0, a, b[0]) * delta_t + random_force_values[0]], dtype=np.float32)
    times = np.array([t], dtype=np.float32)
    # losujemy tau, czyli jak długo kanał jest O/Z
    tau = np.random.exponential(b[1])
    while t < records:
        x = np.append(
        x, (
            x[t - 1] +
            model_force(x[t - 1], a, b[0]) * delta_t +
            random_force_values[t - 1]
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
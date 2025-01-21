import numpy as np
import matplotlib.pyplot as plt
from random import gauss, choice
from scipy.stats import levy_stable
from statsmodels.graphics.tsaplots import plot_acf
import fathon
from fathon import fathonUtils as fu


def random_force_gauss(D, h, records, args, opened_state):
    """Function generates random force values using gaussian distribution.

    Args:
        D (float): Needs to be similar to h in terms of order of magnitude.
        h (float): Needs to be similar to D in terms of order of magnitude.
        records (int): Number of records to generate.
        args (dict): Placeholder for additional arguments.
        opened_state (bool): State of ion channel.

    Returns:
        np.array: Array with random force values.
    """
    return np.array([(h * D) ** 0.5 * gauss(0, 1) for _ in range(records)])  # losowa siła

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

# takes two common parameters and dictionary with 5 parameters needed for levy_stable.rvs function
def random_force_levy(D, h, records, args, opened_state):
    """Function generates random force values using levy_stable distribution.

    Args:
        D (float): Needs to be similar to h in terms of order of magnitude.
        h (float): Needs to be similar to D in terms of order of magnitude.
        records (int): Number of records to generate.
        args (dict): Placeholder for additional arguments. Contains: alpha, beta_closed, beta_opened, location, scale.
        opened_state (bool): State of ion channel.

    Returns:
        np.array: Array with random force values.
    """
    r = levy_stable.rvs(alpha=args['alpha'], beta=args['beta_opened' if opened_state else 'beta_closed'], loc=args['location'], scale=args['scale'], size=records)
    return (h * D) ** 0.5 * r  # random force

def ion_channel_model(a=1, closed=(-1, 6.0), opened=(1, 2.0), D=0.5, delta_t=0.01, records=50000, model_force=model_force_square, random_force=random_force_gauss, **force_params):
    """Function, that generates time series of ion channel model and saves it to csv file.
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

    Returns:
        tuple: Tuple with data and breakpoints.
    """
    generator = np.random.Generator(np.random.PCG64(seed=12345))
    data = []
    breakpoints = []
    t = 0
    # generating first state of ion channel (opened/closed)
    b = choice([closed, opened])
    tau = generator.exponential(b[1])
    random_force_values = random_force(D, delta_t, np.int32(tau//delta_t), force_params, b == opened)
    x = np.array([model_force(0, a, b[0]) * delta_t + random_force_values[0]], dtype=np.float32)
    times = np.array([t], dtype=np.float32)
    data.append([times[0], x[0], b[0]])
    t += 1
    while t < records:
        x = np.append(
        x, (
            x[t - 1] +
            model_force(x[t - 1], a, b[0]) * delta_t +
            random_force_values[t - 1]
            )
        )
        times = np.append(times, t * delta_t)

        data.append([times[t], x[t], b[0]])

        t += 1
        tau -= delta_t
        if(tau <= delta_t):
            breakpoints.append(times[t-1])
            if(b[0] == closed[0]):
                b=opened
            else:
                b=closed
            tau = generator.exponential(b[1])
            tau = 1.0 if tau < 1.0 else tau
            random_force_values = np.append(random_force_values, random_force(D, delta_t, np.int32(tau//delta_t), force_params, b[0] == opened[0]))
            

    data = np.array(data)
    # Temporary naming scheme
    np.savetxt(f'outputs/data_{model_force.__name__}_{random_force.__name__}_{D}_{delta_t}_{list(force_params.values()) if isinstance(force_params, dict) else '_'}.csv', data, delimiter=',', header='time,position,state', fmt=['%.2f', '%e', '%d'])
    return data, breakpoints

def calculate_autocorelation_acf(data, lags=30, title=""):
    """Function calculates and plots autocorrelation function.

    Args:
        data (np.array): Array with data.
        lags (int, optional): Number of lags to calculate. Defaults to 30.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Autocorrelation ' + title)
    plot_acf((data), lags=lags, ax=axs[0], fft=True, title='Unmodified autocorrelation ' + title)
    # plt.show()
    plot_acf(np.diff(data), lags=lags, ax=axs[1], fft=True, title='Modified autocorrelation ' + title)
    fig.savefig(f'outputs/acf_{title}.png')
    plt.show()

def calculate_autocorelation_dfa(data, title=""):
    a = fu.toAggregated(data)

    pydfa = fathon.DFA(a)

    winSizes = fu.linRangeByStep(10, 2000)
    revSeg = True
    polOrd = 3

    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)

    H, H_intercept = pydfa.fitFlucVec()

    plt.plot(np.log(n), np.log(F), 'ro')
    plt.plot(np.log(n), H_intercept+H*np.log(n), 'k-', label='H = {:.2f}'.format(H))
    plt.xlabel('ln(n)', fontsize=14)
    plt.ylabel('ln(F(n))', fontsize=14)
    plt.title('DFA', fontsize=14)
    plt.legend(loc=0, fontsize=14)
    plt.savefig(f'outputs/dfa_{title}.png')
    plt.show()

    limits_list = np.array([[15,2000], [200,1000]], dtype=int)
    list_H, list_H_intercept = pydfa.multiFitFlucVec(limits_list)

    clrs = ['k', 'b', 'm', 'c', 'y']
    stls = ['-', '--', '.-']
    plt.plot(np.log(n), np.log(F), 'ro')
    for i in range(len(list_H)):
        n_rng = np.arange(limits_list[i][0], limits_list[i][1]+1)
        plt.plot(np.log(n_rng), list_H_intercept[i]+list_H[i]*np.log(n_rng),
                clrs[i%len(clrs)]+stls[(i//len(clrs))%len(stls)], label='H = {:.2f}'.format(list_H[i]))
    plt.xlabel('ln(n)', fontsize=14)
    plt.ylabel('ln(F(n))', fontsize=14)
    plt.title('DFA ' + title, fontsize=14)
    plt.legend(loc=0, fontsize=14)
    plt.savefig(f'outputs/dfa_multi_{title}.png')
    plt.show()
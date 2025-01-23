import random
from matplotlib import axes
import numpy as np
import matplotlib.pyplot as plt
from random import choice, seed
from scipy.stats import levy_stable
from statsmodels.graphics.tsaplots import plot_acf
import fathon
from fathon import fathonUtils as fu
import os

def random_force_gauss(D, h, records, args, opened_state, generator):
    """Function generates random force values using gaussian distribution.

    Args:
        D (float): Needs to be similar to h in terms of order of magnitude. Is linked with temperature.
        h (float): Needs to be similar to D in terms of order of magnitude.
        records (int): Number of records to generate.
        args (dict): Placeholder for additional arguments.
        opened_state (bool): State of ion channel.
        generator (numpy.Generator): used for seeded generation

    Returns:
        np.array: Array with random force values.
    """
    
    return (h * D) ** 0.5 * generator.normal(0, 1, size=records)  # losowa siła

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
def random_force_levy(D, h, records, args, opened_state, generator):
    """Function generates random force values using levy_stable distribution.

    Args:
        D (float): Needs to be similar to h in terms of order of magnitude. It's linked with temperature.
        h (float): Needs to be similar to D in terms of order of magnitude.
        records (int): Number of records to generate.
        args (dict): Placeholder for additional arguments. Contains: alpha, beta_closed, beta_opened, location, scale.
        opened_state (bool): State of ion channel.
        generator (numpy.Generator): used for seeded generation

    Returns:
        np.array: Array with random force values.
    """
    r = levy_stable.rvs(alpha=args['alpha'], beta=args['beta_opened' if opened_state else 'beta_closed'], loc=args['location'], scale=args['scale'], size=records, random_state=generator)
    return (h * D) ** 0.5 * r  # random force

def ion_channel_model(a=1, closed=(-1, 6.0), opened=(1, 2.0), D=0.5, delta_t=0.01, records=50000, model_force=model_force_square, random_force=random_force_gauss, takes_prev_vals=True, seed=12345, **force_params):
    """Function, that generates time series of ion channel model and saves it to csv file.
    Args:
        a (_int_, optional): One of the coefficients of force. Defaults to 1.
        closed (_tuple_, optional): Tuple where first item is state of closed ion channel and second is value to be passed to np.random.exponential to generate time the channel is closed. Defaults to (-1, 5000).
        opened (_tuple_, optional): Tuple where first item is state of closed ion channel and second is value to be passed to np.random.exponential to generate time the channel is opened. Defaults to (1, 2500).
        D (_float_, optional): Needs to be similar to delta_t in terms of order of magnitude. Defaults to 0.5.
        delta_t (_float_, optional): Needs to be similar to D in terms of order of magnitude. Defaults to 0.01.
        records (_int_, optional): Number of records to generate.. Defaults to 50000.
        model_force (_func_, optional): Parameter to pass model force. Defaults to model_force_square.
        random_force (_func_, optional): Parameter to pass random force that generates noise. Defaults to random_force_gauss.
        takes_prev_vals (_bool_, optional): idk
        seed (_int_, optional): Seed for np.random.Generator for seeded model generation.
        **force_params: Additional parameters for random force function.

    Returns:
        tuple: Tuple with data and breakpoints.
    """
    generator = np.random.Generator(np.random.PCG64(seed=seed))
    data = []
    breakpoints = []
    t = 0
    # generating first state of ion channel (opened/closed)
    b = choice([closed, opened])
    tau = generator.exponential(b[1])
    random_force_values = random_force(D, delta_t, np.int32(tau//delta_t), force_params, b == opened, generator=generator)
    
    x = np.array([model_force(0, a, b[0]) * delta_t + random_force_values[0]], dtype=np.float32)
    times = np.array([t], dtype=np.float32)
    data.append([times[0], x[0], b[0]])
    t += 1
    no_state_change = True # needed so it doesn't take previous value of previous state (closed and opened are seperated and changes are more acute)
    while t < records:
        x = np.append(
        x, (
            no_state_change * takes_prev_vals * x[t - 1] +
            model_force(x[t - 1], a, b[0]) * delta_t +
            random_force_values[t - 1]
            )
        )
        times = np.append(times, t * delta_t)

        data.append([times[t], x[t], b[0]])
        no_state_change = True
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
            random_force_values = np.append(random_force_values, random_force(D, delta_t, np.int32(tau//delta_t), force_params, b[0] == opened[0], generator=generator))
            no_state_change = False

    data = np.array(data)
    # Temporary naming scheme
    np.savetxt(f'outputs/data_{model_force.__name__}_{random_force.__name__}_{D}_{delta_t}_{list(force_params.values()) if isinstance(force_params, dict) else '_'}.csv', data, delimiter=',', header='time,position,state', fmt=['%.2f', '%e', '%d'])
    return data, breakpoints

def calculate_autocorrelation_acf(data, lags=30, title="test"):
    """Function calculates and plots autocorrelation function.

    Args:
        data (np.array): Array with data.
        lags (int, optional): Number of lags to calculate. Defaults to 30.
        title (string, optional): Directory to save plot to. Defaults to "test"
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Autocorrelation ' + title)
    plot_acf((data), lags=lags, ax=axs[0], fft=True, title='Unmodified autocorrelation')
    # plt.show()
    plot_acf(np.diff(data), lags=lags, ax=axs[1], fft=True, title='Modified autocorrelation')
    if(not os.path.isdir(f"outputs/{title}")):
        os.mkdir(f"outputs/{title}")
    fig.savefig(f'outputs/{title}/acf.png')
    plt.show()

def calculate_autocorrelation_dfa(data, title="test"):
    """ Calculates, shows and saves Detrended Fluctuation Analysis plots with Hurst exponent for autocorrelation. 
    Depending on the value of H it is:
        * H < 0.5 - anti-correlated
        * H around 0.5 - uncorrelated, white noise
        * H > 0.5 - correlated
        * H around - 1/f-noise, pink noise
        * H > 1 - non-stationary, unbounded
        * H around 1.5 - Brownian noise
    Args:
        data (_np.array_): Array with data to calculate autocorelation on.
        title (_str_, optional): Title of figure and directory name to save plots to. Defaults to "test".
    """
    a = fu.toAggregated(data)

    pydfa = fathon.DFA(a)

    winSizes = fu.linRangeByStep(10, 2000)
    revSeg = True
    polOrd = 3

    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)

    H, H_intercept = pydfa.fitFlucVec()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('DFA ' + title)
    axes[0].plot(np.log(n), np.log(F), 'ro')
    axes[0].plot(np.log(n), H_intercept+H*np.log(n), 'k-', label='H = {:.2f}'.format(H))
    axes[0].set_xlabel('ln(n)', fontsize=14)
    axes[0].set_ylabel('ln(F(n))', fontsize=14)
    axes[0].set_title('DFA', fontsize=14)
    axes[0].legend(loc=0, fontsize=14)

    limits_list = np.array([[15,2000], [200,1000]], dtype=int)
    list_H, list_H_intercept = pydfa.multiFitFlucVec(limits_list)

    clrs = ['k', 'b', 'm', 'c', 'y']
    stls = ['-', '--', '.-']
    axes[1].plot(np.log(n), np.log(F), 'ro')
    for i in range(len(list_H)):
        n_rng = np.arange(limits_list[i][0], limits_list[i][1]+1)
        axes[1].plot(np.log(n_rng), list_H_intercept[i]+list_H[i]*np.log(n_rng),
                clrs[i%len(clrs)]+stls[(i//len(clrs))%len(stls)], label='H = {:.2f}'.format(list_H[i]))
    axes[1].set_xlabel('ln(n)', fontsize=14)
    axes[1].set_ylabel('ln(F(n))', fontsize=14)
    axes[1].set_title('DFA', fontsize=14)
    axes[1].legend(loc=0, fontsize=14)
    if(not os.path.isdir(f"outputs/{title}")):
        os.mkdir(f"outputs/{title}")
    plt.savefig(f'outputs/{title}/dfa.png')
    plt.show()
import stat
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from scipy.fft import next_fast_len
from scipy.stats import levy_stable
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
import fathon
from fathon import fathonUtils as fu
import os
import numpy as np
from ipywidgets import FloatSlider, IntSlider, Dropdown, SelectionSlider, VBox, Button, Checkbox
from IPython.display import display
from matplotlib import colors

import multiprocessing


def factor(number):
    """Function that calculates factors of a number.

    Parameters
    ----------
    number : int
        Number to calculate factors of.

    Returns
    -------
    ndarray
        Array with factors of a number.
    """
    products = []
    product = 2
    while product*product <= number:
        while number % product == 0:
            products.append(product)
            number = number//product
        product += 1    
    return np.array(products)[::-1]

class IonChannel:
    """
    IonChannel
    ----------
    Class with methods to generate and analyse Ion Channel time series. 
    
    Attributes
    ----------
    data : ndarray
        Holds data generated by method ``_generate_data``.
    data_transposed : ndarray
        Holds transposed ``data``.
    breakpoints : ndarray
        Holds points at which Ion Channel changed state.
    dwell_times : ndarray
        Holds dwell times of Ion Channel. These times are the time the Ion Channel stayed in one state.

    Methods
    -------
    _generate_data(random_force='Gauss'):
        Generates Ion Channel time series data
    """

    def __init__(self, a=1000.0, closed=(-38.0, 40.0), opened=(-34.0, 15.0), D=100.0, delta_t=0.0001, records=50000, takes_prev_vals=True, seed=12345, **force_params):
        """Constructor of ``IonChannel`` class

        Parameters
        ----------
        a : float, optional
            a coefficient used in ``__model_force_square``, by default 100
        closed : tuple, optional
            ``closed[0]`` has value of current in [pA] of when channel is closed.

            ``closed[1]`` has value of average time the channel is closed. By default (-1, 48.0)
        opened : tuple, optional
            ``opened[0]`` has value of current in [pA] of when channel is opened.
            
            ``opened[1]`` has value of average time the channel is opened. By default (1, 12.0)
        D : float, str, optional
            String representing how strong is the noise. Values are:
            - ``VL`` for very low noise. Data is easily readable.
            - ``L`` for low noise. Data is readable.
            - ``M`` for medium noise. Data is moderately readable.
            - ``H`` for high noise. Data is unreadable.
            Float value can be used to set custom noise level.
            By default 100.0.
        delta_t : float, optional
            Time series spacing. Should be magnification of 10 to negative power, by default 0.0001
        records : int, optional
            Number of records to generate, by default 50000
        takes_prev_vals : bool, optional
            Was used for testing. Keep at ``True``, by default True
        seed : int, optional
            Used for seeded generation. Creates ``np.random.Generator(np.random.PCG64(seed=seed))``, by default 12345
        force_params : dict, optional
            Parameters for Levy distribution and for asymetrical model force, by default {}.
        """
        self.__a = a
        self.__closed = list(closed)
        self.__opened = list(opened)
        self.__closed[1] = self.__closed[1]*delta_t*100
        self.__opened[1] = self.__opened[1]*delta_t*100
        self.__LUT_D = {'VL' : 50.0, 'L' : 100.0, 'M' : 200.0, 'H' : 1000.0}
        if isinstance(D, float):
            self.__D = D
        else:
            self.__D = self.__LUT_D[D]

        self.__delta_t = delta_t
        self.__records = records
        self.__takes_prev_vals = takes_prev_vals
        self.__force_params = force_params
        # generator for state and how long its in the state
        self.__generator1 = np.random.Generator(np.random.PCG64(seed=seed))
        # generator for random force
        self.__generator2 = np.random.Generator(np.random.PCG64(seed=seed))
        
        self.data = []
        self.breakpoints = []

    def __random_force_gauss(self, records):
        """Function generates random force values using standard gaussian distribution.

        Parameters
        ----------
        records : _int_
            Number of records to generate.

        Returns
        -------
        ndarray : 
            Array with random force values.
        """
        
        return self.__D ** 0.5 * (2 * self.__delta_t) ** 0.5 * self.__generator2.normal(0, 1, size=records)
    
    def __random_force_levy(self, records):
        """Function generates random force values using ``levy_stable`` distribution.

        Parameters
        ----------
        records : int
            Number of records to generate.

        Returns
        -------
        ndarray: 
            Array with random force values.
        """
        args = self.__force_params
        if self._opened_larger:
            r = levy_stable.rvs(alpha=args['alpha'], beta=args['beta'] if self.__opened_state else -args['beta'], loc=0.0, scale=args['scale'], size=records, random_state=self.__generator2)
        else:
            r = levy_stable.rvs(alpha=args['alpha'], beta=-args['beta'] if self.__opened_state else args['beta'], loc=0.0, scale=args['scale'], size=records, random_state=self.__generator2)
        return (self.__D ** 0.5) * (2 * self.__delta_t) ** 0.5 * r  # random force
    
    def __model_force_square(self, x, b):
        """Function calculates force value using square function.

        Parameters
        ---------
        x : float 
            Position value.
        b : float 
            State of the ion channel.

        Returns
        -------
        float: 
            Force value at x.
        """
        return -self.__a*(x - b)

    def __model_force_asymetrical(self, x, b):
        """Function calculates force value with asymetrical potential.

        Parameters
        ----------
        x : float
            Position value.
        b : float
            State of the ion channel.            

        Returns
        -------
        float
            Float value at x.
        """
        k = self.__force_params['k']
        if self._opened_larger:
            k = k if self.__opened_state else -k
        else:
            k = -k if self.__opened_state else k
        return self.__model_force_square(x, b) * np.e**(k*(x-b))
    
    def __model_force_piecewise(self, x, b):
        """Function calculates force value with piecewise potential.
        """
        if x==b:
            return 0.0
        if self._opened_larger:
            if x < self.__closed[0]:
                return -self.__model_force_square(x, b)
            if x > self.__opened[0]:
                return self.__model_force_square(x, b)
        else:
            if x < self.__opened[0]:
                return -self.__model_force_square(x, b)
            if x > self.__closed[0]:
                return self.__model_force_square(x, b)
        k = self.__force_params['k']
        return -k*self.__a*(x-b)*(x-self.__closed[0]) if self.__opened_state else self.__a*(x-b)*(x-self.__opened[0])

    # def __model_force_piecewise_simple(self, x, b):
    #     if np.abs(x-b) < 0.1:
    #         return 0
    #     if x > b:
    #         return -self.__a
    #     else:
    #         return self.__a

    def __model_force_piecewise_simple(self, x, b):
        if np.abs(x-b) < 0.1:
            return 0.0
        if self._opened_larger:
            # przypadek gdy otwarty większy
            if b == self.__closed[0]:
                # przypadek gdy stan jest zamknięty
                if x > self.__closed[0] and x <= self.__opened[0]:
                    return -self.__a
                if x > self.__opened[0]:
                    return self.__model_force_square(x, self.__opened[0] - 1)
                if x < self.__closed[0]:
                    return self.__model_force_square(x, self.__closed[0] + 1)
            else:
                # przypadek gdy stan jest otwarty
                if x >= self.__closed[0] and x < self.__opened[0]:
                    return self.__a
                if x > self.__opened[0]:
                    return self.__model_force_square(x, self.__opened[0] - 1)
                if x < self.__closed[0]:
                    return self.__model_force_square(x, self.__closed[0] + 1)
        else:
            # przypadek gdy zamknięty większy
            if b == self.__closed[0]:
                # przypadek gdy stan jest zamknięty
                if x >= self.__opened[0] and x < self.__closed[0]:
                    return self.__a
                if x > self.__closed[0]:
                    return self.__model_force_square(x, self.__closed[0] - 1)
                if x < self.__opened[0]:
                    return self.__model_force_square(x, self.__opened[0] + 1)
            else:
                # przypadek gdy stan jest otwarty
                if x > self.__opened[0] and x <= self.__closed[0]:
                    return -self.__a
                if x > self.__closed[0]:
                    return self.__model_force_square(x, self.__closed[0] - 1)
                if x < self.__opened[0]:
                    return self.__model_force_square(x, self.__opened[0] + 1)

    # def __transition(self, b1, b2, t, T):
    #     return b1 + (b2 - b1) * t / T

    # def __model_force_smooth(self, x, t, T, previous_state, desired_state):
    #     """Function calculates force value with smooth transition.
    #     """
    #     b = self.__transition(previous_state, desired_state, t, T)
    #     print("x", x)
    #     print("b", b)
    #     print(self.__model_force_square(x, b))
    #     return self.__model_force_square(x, b)


    
    def _generate_data(self, model_force="Piecewise_simple", random_force='Gauss'):
        """Function, that generates time series of ion channel model and saves it to csv file.

        Parameters
        ----------
        model_force : str
            Choice of model function.
                - ``Square`` for ``__model_force_square``
                - `asymetrical` for ``__model_force_asymetrical``
        random_force : str
            Choice of noise random function. 
                - ``Gauss`` for ``__random_force_gauss``
                - ``Levy`` for ``__random_force_levy``
        """
        if random_force.lower() == str.lower('Gauss'):
            self.__random_force = self.__random_force_gauss
        else:
            self.__random_force = self.__random_force_levy

        match model_force.lower() :
            case "square":
                self.__model_force = self.__model_force_square
            case "asymetrical":
                self.__model_force = self.__model_force_asymetrical
            case "piecewise":
                self.__model_force = self.__model_force_piecewise
            case "piecewise_simple":
                self.__model_force = self.__model_force_piecewise_simple
        t = 0
        self.dwell_times = []
        # generating first state of ion channel (opened/closed)
        b = self.__generator1.choice([self.__closed, self.__opened])
        tau = self.__generator1.exponential(b[1])
        # print(tau)
        self.dwell_times.append(tau)
        self._opened_larger = self.__opened[0] > self.__closed[0]
        self.__opened_state = b[0] == self.__opened[0]
        random_force_values = self.__random_force(np.int32(tau//self.__delta_t))
        x = np.array([b[0]], dtype=np.float32)
        times = np.linspace(0, self.__records*self.__delta_t, self.__records, endpoint=False)
        self.data.append([times[0], x[0], b[0]])
        t += 1
        while t < self.__records:
            new_x = (
                self.__takes_prev_vals * x[t - 1] +
                # self.__model_force(x[t - 1], b[0]) * self.__delta_t +
                self.__model_force(x[t - 1], b[0]) * self.__delta_t +
                random_force_values[t]
                )
            x = np.append(x, new_x)
            self.data.append([times[t], x[t], b[0]])
            t += 1
            tau -= self.__delta_t
            if(tau - self.__delta_t < self.__delta_t):
                # print("Wow breakpoint")
                self.breakpoints.append(times[t-1])
                self.__opened_state = b[0] == self.__opened[0]
                if self.__opened_state:
                    b=self.__closed
                else:
                    b=self.__opened
                tau = self.__generator1.exponential(b[1])
                self.dwell_times.append(tau)
                random_force_values = np.append(random_force_values, self.__random_force(np.int32(tau//self.__delta_t)))


        self.data = np.array(self.data)
        self.breakpoints = np.array(self.breakpoints)        
        self.data_transposed = self.data.T
        save_file = os.path.join(os.getcwd(), 'outputs', f'data_{random_force}_a{self.__a}_D{self.__D}_o{self.__opened[0]}_c{self.__closed[0]}_delta_t{self.__delta_t}_{list(self.__force_params.values()) if isinstance(self.__force_params, dict) else '_'}.csv')
        np.savetxt(save_file, self.data, delimiter=',', header='time,position,state', fmt=['%e', '%e', '%d'])

    def __calculate_ranges(self):
        """Calculates ranges of data.
        """
        LUT_threshold = {10.0: 1.0, 50.0: 2.0, 100.0 : 3.0, 200.0 : 4.0, 500.0 : 7.0, 1000 : 9.0}
        minimum = np.min(self.data_transposed[1])
        maximum = np.max(self.data_transposed[1])
        
        if self._opened_larger:
            self.__bottom = self.__closed[0] - LUT_threshold[self.__D] if self.__closed[0] - LUT_threshold[self.__D] > minimum else minimum
            self.__top = self.__opened[0] + LUT_threshold[self.__D] if self.__opened[0] + LUT_threshold[self.__D] < maximum else maximum
            
        else:
            self.__bottom = self.__opened[0] - LUT_threshold[self.__D] if self.__opened[0] - LUT_threshold[self.__D] > minimum else minimum
            self.__top = self.__closed[0] + LUT_threshold[self.__D] if self.__closed[0] + LUT_threshold[self.__D] < maximum else maximum
            

    def plot_time_series(self, ax : plt.Axes, title='Generated model', plot_breakpoints=False, plot_zoomed=False):
        """Plots generated time series.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
            Subplot ax to put plot into.
        title : str, optional
            Title of plot, by default 'Generated model'
        plot_breakpoints : bool
            True if lines on breakpoints should be plotted, by default False.
        plot_zoomed : bool
            True if zoomed in plot should be plotted, by default False.
        
        Raises
        ------
        AssertionError
            With message "Data wasn't generated"

        Returns
        -------
        matplotlib.pyplot.Figure
            Subplot with zoomed. None if ``plot_zoomed`` is False.
        """
        assert len(self.data_transposed) > 0, "Data wasn't generated"
        ax.set_title(title)
        ax.plot(self.data_transposed[0], self.data_transposed[1])
        self.__calculate_ranges()
        ax.set_ylim(bottom=self.__bottom, top=self.__top)
        if plot_breakpoints:
            ax.vlines(x=self.breakpoints, ymin=np.min(self.data_transposed[1]), ymax=np.max(self.data_transposed[1]), color='red', linestyle='--', alpha=0.6)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Current [pA]")

        if plot_zoomed:
            sub_fig, sub_ax = plt.subplots()
            sub_fig.set_size_inches(12, 6)
            sub_fig.suptitle('Zoomed time series')
            sub_fig.tight_layout()
            sub_ax.set_xlabel("Time [s]")
            sub_ax.set_ylabel("Current [pA]")
            sub_ax.plot(self.data_transposed[0][900:1000], self.data_transposed[1][900:1000], 'go-')
        else:
            sub_fig = None
        
        return sub_fig

    
    def plot_time_series_histogram(self, ax, bins=100):
        """Plots histogram of time series with specified amount of ``bins``.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
            Subplot ax to put plot into.
        bins : int, optional
            Number of bins, by default 100.
        """
        N, bins, patches = ax.hist(self.data_transposed[1], bins=bins, range=(self.__bottom, self.__top))
        # range=(self.__bottom, self.__top)
        fracs = N / N.max()
        norm = colors.Normalize(fracs.min(), fracs.max())
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.cividis(norm(thisfrac))
            thispatch.set_facecolor(color)
        ax.set_title('Histogram of time series')
        
        

    def calculate_hurst_exponent(self, data):
        
        N = len(data)
        if N < 20:
            raise ValueError("Time series is too short! input series ought to have at least 20 samples!")

        len_of_subsets = np.cumprod(factor(N))[::-1]
        R_S_dict = []
        for k in len_of_subsets:
            R,S = 0, 0
            # split data into subsets
            subset_list = np.array([data[i:i+k] for i in range(0, N, k)])
            n = len(subset_list)
            # calc mean of every subset
            mean_list = np.array([np.mean(x) for x in subset_list])
            # calc cumsum of every subset
            for i in range(n):
                cumsum_list = np.cumsum(subset_list[i]-mean_list[i])
                R += max(cumsum_list) - min(cumsum_list)
                S += np.std(subset_list[i])
            R_S_dict.append({"R":R/len(subset_list),"S":S/len(subset_list),"n":k})
        
        log_R_S = []
        log_n = []
        # print(R_S_dict)
        for i in range(len(R_S_dict)):
            R_S = (R_S_dict[i]["R"]+np.spacing(1)) / (R_S_dict[i]["S"]+np.spacing(1))
            log_R_S.append(np.log(R_S))
            log_n.append(np.log(R_S_dict[i]["n"]))
        
        Hurst_exponent = np.polynomial.Polynomial.fit(log_n,log_R_S,1).convert().coef[1]
        # Hurst_exponent = np.polyfit(log_n,log_R_S,1)[0]
        return Hurst_exponent

    def calculate_autocorrelation_acf(self, data, fig, ax, lags=100):
        """Function calculates and plots autocorrelation function.
        
        Parameters
        ----------
        data : ndarray
            1D array containing data to calculate autocorrelation on.
        fig : matplotlib.pyplot.Figure
            Figure on which autocorrelation is plotted.
        ax : matplotlib.pyplot.Axes
            Subplot ax to put plot into.
        lags : int, optional
            Number of lags to calculate. Defaults to 30.

        Examples:
        ---------
        >>> import numpy as np
        >>> from ion_channel import IonChannel
        >>> ic = IonChannel()
        >>> data = np.random.random(10000)
        >>> ic.calculate_autocorrelation_acf(data)
        """
        fig = plot_acf(data, ax=ax, lags=lags, fft=True, title='FFT')
        return fig, ax

    def dfa(self, data):
        a = fu.toAggregated(data)
        pydfa = fathon.DFA(a)
        winSizes = fu.linRangeByStep(5, len(a), step=5)
        
        revSeg = True
        polOrd = 3

        n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
        max_limit = np.log(winSizes[-1])
        mid_point = winSizes[int(np.round(np.e**(max_limit//2), decimals=0))]
        limits_list = np.array([[winSizes[10], mid_point], [mid_point, winSizes[-1]], [winSizes[10], winSizes[-1]]], dtype=int)
        list_alpha, list_alpha_intercept = pydfa.multiFitFlucVec(limits_list)

        return n, F, list_alpha, list_alpha_intercept, limits_list

    def plot_autocorrelation_dfa(self, data, fig, ax, stationarity=False):
        """
        Calculates, shows and saves Detrended Fluctuation Analysis plots with Hurst exponent for autocorrelation. 
        Depending on the value of H it is:
        * alpha < 0.5 - anti-correlated
        * alpha around 0.5 - uncorrelated, white noise
        * alpha > 0.5 - correlated
        * alpha around 1 - 1/f-noise, pink noise
        * alpha > 1 - non-stationary, unbounded
        * alpha around 1.5 - Brownian noise
        ----------    

        Parameters
        ----------
        data : ndarray
            1D array containing data to calculate autocorrelation on.
        fig : matplotlib.pyplot.Figure
            Figure on which autocorrelation is plotted.
        ax : matplotlib.pyplot.Axes
            Subplot ax to put plot into.

        Examples:
        ---------
        >>> import numpy as np
        >>> from ion_channel import IonChannel
        >>> ic = IonChannel()
        >>> data = np.random.random(10000)
        >>> ic.plot_autocorrelation_dfa(data)
        """
        if stationarity:
            # check if time series is stationary
            stationary = False
            result = adfuller(data)
            if result[1] > 0.05:
                stationary = True
        
        n, F, list_alpha, list_alpha_intercept, limits_list = self.dfa(data)

        clrs = ['k', 'b', 'm', 'c', 'y']
        stls = ['-', '--', '.-']
        ax.plot(np.log(n), np.log(F), 'ro')
        for i in range(len(list_alpha)):
            n_rng = np.arange(limits_list[i][0], limits_list[i][1]+1)
            ax.plot(np.log(n_rng), list_alpha_intercept[i]+list_alpha[i]*np.log(n_rng),
                    clrs[i%len(clrs)]+stls[(i//len(clrs))%len(stls)], label=(r'$\alpha$' + ' = {:.2f}'.format(list_alpha[i])))
        ax.set_xlabel('ln(n)', fontsize=14)
        ax.set_ylabel('ln(F(n))', fontsize=14)
        ax.set_title(('Stationary' if stationary else 'Non-stationary ') if stationarity else ('') + 'DFA' , fontsize=14)
        ax.legend(loc=0, fontsize=14)
        return fig, ax



    def save_figure(self, fig : plt.Figure, title : str, name = "figure", with_subfigures=True):
        """Saves ``fig`` into folder *outputs2* subfolder ``title`` and file name ``name`` .

        Parameters
        ----------
        fig : matplotlib.pyplot.Figure
            Can be figure generated by: 
            - ```calculate_autocorrelation_acf```
            - ```plot_autocorrelation_dfa```
            - any figure
        title : str
            Folder named by what data was generated.
        name : str
            Name of figure "``name``.png".
        with_subfigures : bool
            Use only if ``fig`` is a subplot with dimensions 2x2.
        """
        path = os.path.join(os.getcwd(), 'outputs2')
        if not os.path.exists(path):
            os.mkdir(path)
        path_with_subfolder = os.path.join(path, title)
        if not os.path.exists(path_with_subfolder):
            os.mkdir(path_with_subfolder)
        fig.savefig(os.path.join(path_with_subfolder, f"{name}.png"))
        if with_subfigures:
            transformation = fig.transFigure - fig.dpi_scale_trans
            fig.savefig(os.path.join(path_with_subfolder, 'sub_figure1.png'), bbox_inches=mtransforms.Bbox([[0, 0], [0.5, 0.495]]).transformed(transformation))
            fig.savefig(os.path.join(path_with_subfolder, 'sub_figure2.png'), bbox_inches=mtransforms.Bbox([[0.5, 0], [1.0, 0.5]]).transformed(transformation))
            fig.savefig(os.path.join(path_with_subfolder, 'sub_figure3.png'), bbox_inches=mtransforms.Bbox([[0, 0.485], [0.5, 0.978]]).transformed(transformation))
            fig.savefig(os.path.join(path_with_subfolder, 'sub_figure4.png'), bbox_inches=mtransforms.Bbox([[0.5, 0.5], [1, 0.978]]).transformed(transformation))

class InteractiveIonChannel():
    """
    InteractiveIonChannel
    ---------------------
    Helper class for ``IonChannel``. Creates easy to manage interactive widget for ``IonChannel`` creation.

    Examples
    --------
    >>> from ion_channel import InteractiveIonChannel
    >>> interactive_ion_channel = InteractiveIonChannel()
    >>> interactive_ion_channel.interact()
    """
    def __init__(self):
        """Constructor of InteractiveIonChannel class
        """
        self.__a_slider = FloatSlider(min=0.0, max=5000.0, step=1, value=1000.0, description='a')
        self.__closed_0_slider = IntSlider(min=-50, max=50, step=1, value=-38, description='Closed value')
        self.__closed_1_slider = FloatSlider(min=0.0, max=1000.0, step=0.1, value=30.0, description='Closed avg time(scaled by delta_t)')
        self.__opened_0_slider = IntSlider(min=-50, max=50, step=1, value=-34, description='Opened value')
        self.__opened_1_slider = FloatSlider(min=0.0, max=1000.0, step=0.1, value=3.0, description='Opened avg time(scaled by delta_t)')
        self.__D_slider = FloatSlider(min=0.00, max=5000.00, step=0.01, value=100.0, description='D')
        self.__delta_t_slider = SelectionSlider(
            options=[10**-i for i in range(3, 6)],
            value=0.0001,
            description='Delta t',
        )
        self.__records_slider = IntSlider(min=1000, max=100000, step=1000, value=50000, description='Records')

        self.__random_force_dropdown = Dropdown(
            options = ['Gauss', 'Levy'],
            value = 'Gauss',
            description = 'Random Force',
        )

        self.__autocorrelation_dropdown = Dropdown(
            options = ['DFA', 'FFT', 'All'],
            value = 'All',
            description = 'Autocorrelation method'
        )

        self.__force_dropdown = Dropdown(
            options = ['Square', 'Asymetrical', 'Piecewise', 'Piecewise_simple'],
            value = 'Piecewise_simple',
            description = 'Model force'
        )

        self.__fft_lags = IntSlider(min=30, max=200, step=10, value=100, description='FFT lags')
        self.__takes_previous = Checkbox(value=True, description='Takes previous values')
        self.__draw_vlines_at_breakpoint = Checkbox(value=False, description='Draw breakpoints')
        self.__seed_select = IntSlider(min=0, max=99999, value=12345, step=1, description='Seed')
        self.__force_params_box = VBox()
        

    def __ion_channel_interactive(self):
        """
        Generates an interactive widget for the ion channel model.
        """
        closed = (self.__closed_0_slider.value, self.__closed_1_slider.value)
        opened = (self.__opened_0_slider.value, self.__opened_1_slider.value)
        self.ion_channel = IonChannel(
            a=self.__a_slider.value,
            closed=closed,
            opened=opened,
            D=self.__D_slider.value,
            delta_t=self.__delta_t_slider.value,
            records=self.__records_slider.value,
            takes_prev_vals=self.__takes_previous.value,
            seed=int(self.generator.random()*100000),
            **self.__force_params
        )
        self.ion_channel._generate_data(self.__force_dropdown.value, self.__random_force_dropdown.value)
    
    def __update_force_params(self, *args):
        """
        Updates the force_params_box based on the selected random force.
        """
        random_force_type = self.__random_force_dropdown.value
        force_type = self.__force_dropdown.value
        force_params = []
        if random_force_type.lower() == str.lower('Levy'):
            alpha = FloatSlider(min=1.5, max=1.99, step=0.01, value=1.9, description='alpha')
            beta = FloatSlider(min=0.0, max=1.0, step=0.01, value=0.9, description='beta')
            scale = FloatSlider(min=0, max=100, step=0.1, value=1.5, description='scale')
            force_params = [alpha, beta, scale]
        else: 
            force_params = []

        if force_type.lower() == str.lower('asymetrical') or force_type.lower() == str.lower('piecewise'):
            k = FloatSlider(min=0, max=10, step=0.01, value=1.0, description='k')
            force_params.append(k)
        self.__force_params_box.children = force_params

    def interact(self):
        """Main function to call on ``InteractiveIonChannel`` class. 

        Creates interactive ``ipywidget``, that user can initialize ``IonChannel`` with.
        """
        self.__force_dropdown.observe(self.__update_force_params, names='value')
        self.__random_force_dropdown.observe(self.__update_force_params, names='value')

        display(
            self.__seed_select,
            self.__a_slider,
            self.__closed_0_slider,
            self.__closed_1_slider,
            self.__opened_0_slider,
            self.__opened_1_slider,
            self.__D_slider,
            self.__delta_t_slider,
            self.__records_slider,
            self.__takes_previous,
            self.__autocorrelation_dropdown,
            self.__fft_lags,
            self.__random_force_dropdown,
            self.__force_dropdown,
            self.__force_params_box,
            self.__draw_vlines_at_breakpoint
        )
        run_button = Button(description="Run Model")
        run_button.on_click(self.__on_button_click)
        display(run_button)

        run_button_test = Button(description="Run Model (Test)")
        run_button_test.on_click(self.__on_button_click)
        display(run_button_test)
        
    def __on_click_event(self, D):
        """Helper method for ``__on_button_click__``. Generates data(plots) according to users input in ``interact``.
        
        Parameters
        ----------
        D : float
            Value of coefficient D. Represents noise level.
        """
        fig, axs = plt.subplots(2, 2, constrained_layout=True)
        fig.set_size_inches(16, 12)
        sub_fig = self.ion_channel.plot_time_series(axs[0][0], plot_breakpoints=self.__draw_vlines_at_breakpoint.value, plot_zoomed=True)
        self.ion_channel.plot_time_series_histogram(axs[0][1])
        match self.__force_dropdown.value:
            case "Square":
                name = "Square_"
            case "Asymetrical":
                name = f"Asymetrical_k{self.__force_params['k']}_"
            case "Piecewise":
                name = "Piecewise_"
            case "Piecewise_simple":
                name = "Piecewise_simple_"
        match self.__random_force_dropdown.value:
            case "Gauss":
                name += f"{self.__random_force_dropdown.value}_D{D}_a{self.__a_slider.value}_{self.__seed_select.value}"
            case "Levy":
                name += f"{self.__random_force_dropdown.value}_D{D}_a{self.__a_slider.value}_alpha{self.__force_params["alpha"]}_beta{self.__force_params["beta"]}_scale{self.__force_params["scale"]}_{self.__seed_select.value}"
        

        data = self.ion_channel.data_transposed[1] # because self.data is shape (records, 3)
        fig.suptitle(name)
        match self.__autocorrelation_dropdown.value:
            case 'DFA': 
                fig, axs[1][0] = self.ion_channel.plot_autocorrelation_dfa(data, fig, axs[1][0])
            case 'FFT':
                fig, axs[1][0]  = self.ion_channel.calculate_autocorrelation_acf(data, fig, axs[1][0], lags=self.__fft_lags.value)
            case _:
                fig, axs[1][0] = self.ion_channel.calculate_autocorrelation_acf(data, fig, axs[1][0], lags=self.__fft_lags.value)
                fig, axs[1][1] = self.ion_channel.plot_autocorrelation_dfa(data, fig, axs[1][1])

        self.ion_channel.save_figure(fig, title=name)
        self.ion_channel.save_figure(sub_fig, title=name, name='zoomed_time_series', with_subfigures=False)

    def __multi_on_click_event(self, index):
        data = self.ion_channel.data_transposed[1]
        self.__fig_autocorr, self.__axs_autocorr[index][0] = self.ion_channel.calculate_autocorrelation_acf(data, self.__fig_autocorr, self.__axs_autocorr[index][0], lags=self.__fft_lags.value)
        self.__fig_autocorr, self.__axs_autocorr[index][1] = self.ion_channel.plot_autocorrelation_dfa(data, self.__fig_autocorr, self.__axs_autocorr[index][1])
        
    def __alpha_on_click_event(self, D):
        self.__ion_channel_interactive()
        return self.ion_channel.dfa(self.ion_channel.data_transposed[1])[2]

    @staticmethod
    def multiprocessed_worker(args):
        D, nr_of_tests, seed, force_params, force_dropdown_value, random_force_dropdown_value, delta_t_slider_value, records_slider_value, takes_previous_value, closed, opened, a_slider_value = args
        alpha_low = 0
        alpha_high = 0
        alpha_all = 0
        for _ in range(nr_of_tests):
            ion_channel = IonChannel(
                a=a_slider_value,
                closed=closed,
                opened=opened,
                D=D,
                delta_t=delta_t_slider_value,
                records=records_slider_value,
                takes_prev_vals=takes_previous_value,
                seed=seed,
                **force_params
            )
            ion_channel._generate_data(force_dropdown_value, random_force_dropdown_value)
            alpha_list = ion_channel.dfa(ion_channel.data_transposed[1])[2]
            alpha_low += alpha_list[0]
            alpha_high += alpha_list[1]
            alpha_all += alpha_list[2]
        return alpha_low, alpha_high, alpha_all
    
    def __on_button_click(self, b : Button):
        """
        Callback for the 'Run Model'/'Run Model (test)' button.
        """
        self.__force_params = {child.description: child.value for child in self.__force_params_box.children}
        if (b.description == 'Run Model'):
            self.generator = np.random.Generator(np.random.PCG64(seed=self.__seed_select.value))
            self.__ion_channel_interactive()
            self.__on_click_event(self.__D_slider.value)
        else:
            alpha_low = 0
            alpha_high = 0
            alpha_all = 0
            self.generator = np.random.Generator(np.random.PCG64(seed=self.__seed_select.value))
            nr_of_tests = 108
            core_count = multiprocessing.cpu_count()
            batch_for_core = nr_of_tests // core_count
            noise_dict = dict([])
            D_list = [10.0, 50.0, 100.0, 500.0]
            for noise in ["Gauss", "Levy"]:
                self.__random_force_dropdown.value = noise
                for D in D_list:
                    alpha_dict = dict([])
                    args_list = [
                        (D, batch_for_core, self.__seed_select.value, self.__force_params, self.__force_dropdown.value, self.__random_force_dropdown.value, self.__delta_t_slider.value, self.__records_slider.value, self.__takes_previous.value, (self.__closed_0_slider.value, self.__closed_1_slider.value), (self.__opened_0_slider.value, self.__opened_1_slider.value), self.__a_slider.value)
                    ]
                    with multiprocessing.Pool(core_count) as p:
                        results = p.map(self.multiprocessed_worker, args_list)
                    for result in results:
                        alpha_low += result[0]
                        alpha_high += result[1]
                        alpha_all += result[2]
                    alpha_low /= nr_of_tests
                    alpha_high /= nr_of_tests
                    alpha_all /= nr_of_tests
                    alpha_dict[D] = [alpha_low, alpha_high, alpha_all]
                noise_dict[noise] = alpha_dict
            with open('alpha.csv', 'w') as f:
                f.write('Noise,D,alpha_low,alpha_high,alpha_all\n')
                for noise, alpha_dict in noise_dict.items():
                    for D, alphas in alpha_dict.items():
                        f.write(f'{noise},{D},{alphas[0]},{alphas[1]},{alphas[2]}\n')
import matplotlib.pyplot as plt
from random import choice
from scipy.stats import levy_stable
from statsmodels.graphics.tsaplots import plot_acf
import fathon
from fathon import fathonUtils as fu
import os
import numpy as np
from ipywidgets import FloatSlider, IntSlider, Dropdown, SelectionSlider, VBox, Button, Checkbox
from IPython.display import display

class IonChannel:
    
    def __init__(self, a=1, closed=(-1, 6.0), opened=(1, 2.0), D=0.5, delta_t=0.01, records=50000, takes_prev_vals=True, seed=12345, **random_force_params):
        self.a = a
        self.closed = closed
        self.opened = opened
        self.D = D
        self.delta_t = delta_t
        self.records = records
        self.takes_prev_vals = takes_prev_vals
        self.random_force_params = random_force_params
        self.generator = np.random.Generator(np.random.PCG64(seed=seed))

        

    def random_force_gauss(self, records, opened_state):
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
        
        return (self.delta_t * self.D) ** 0.5 * self.generator.normal(0, 1, size=records)  # losowa siła
    
    def model_force_square(self, x, b):
        """Function calculates force value using square function.

        Args:
            x (float): Position value.
            a (float): Coefficient.
            b (float): Coefficient.

        Returns:
            float: Force value at x.
        """
        return -self.a*(x - b)
    
    # takes two common parameters and dictionary with 5 parameters needed for levy_stable.rvs function
    def random_force_levy(self, records, opened_state):
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
        args = self.random_force_params
        r = levy_stable.rvs(alpha=args['alpha'], beta=args['beta_opened' if opened_state else 'beta_closed'], loc=args['location'], scale=args['scale'], size=records, random_state=self.generator)
        return (self.delta_t * self.D) ** 0.5 * r  # random force
    
    def ion_channel_model(self, random_force='Gauss'):
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
        if random_force.lower() == str.lower('Gauss'):
            self.random_force = self.random_force_gauss
        else:
            self.random_force = self.random_force_levy

        data = []
        self.breakpoints = []
        t = 0
        # generating first state of ion channel (opened/closed)
        b = choice([self.closed, self.opened])
        tau = self.generator.exponential(b[1])
        random_force_values = self.random_force(np.int32(tau//self.delta_t), b == self.opened)
        
        x = np.array([self.model_force_square(0, b[0]) * self.delta_t + random_force_values[0]], dtype=np.float32)
        times = np.array([t], dtype=np.float32)
        data.append([times[0], x[0], b[0]])
        t += 1
        no_state_change = True # needed so it doesn't take previous value of previous state (closed and opened are seperated and changes are more acute)
        while t < self.records:
            x = np.append(
            x, (
                no_state_change * self.takes_prev_vals * x[t - 1] +
                self.model_force_square(x[t - 1], b[0]) * self.delta_t +
                random_force_values[t - 1]
                )
            )
            times = np.append(times, t * self.delta_t)

            data.append([times[t], x[t], b[0]])
            no_state_change = True
            t += 1
            tau -= self.delta_t
            if(tau <= self.delta_t):
                self.breakpoints.append(times[t-1])
                if(b[0] == self.closed[0]):
                    b=self.opened
                else:
                    b=self.closed
                tau = self.generator.exponential(b[1])
                tau = 1.0 if tau < 1.0 else tau
                random_force_values = np.append(random_force_values, self.random_force(np.int32(tau//self.delta_t), b[0] == self.opened[0]))
                no_state_change = False

        self.data = np.array(data).T
        # Temporary naming scheme
        np.savetxt(f'outputs/data_{random_force}_{self.D}_{self.delta_t}_{list(self.random_force_params.values()) if isinstance(self.random_force_params, dict) else '_'}.csv', data, delimiter=',', header='time,position,state', fmt=['%.2f', '%e', '%d'])
        return self.data, self.breakpoints
    
    def plot_time_series(self, figsize=(10,5), title='Generated model'):
        assert self.data.size > 0, "Data wasn't generated"
        
        plt.clf()  
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.plot(self.data[0], self.data[1])
        plt.xlabel("Time [s]")
        plt.ylabel("Current [mA]")
        plt.show()

    def plot_time_series_histogram(self, bins=100):
        plt.hist(self.data[1], bins=bins)
        plt.show()

    def calculate_autocorrelation_acf(self, lags=30, title="test"):
        """Function calculates and plots autocorrelation function.

        Args:
            data (np.array): Array with data.
            lags (int, optional): Number of lags to calculate. Defaults to 30.
            title (string, optional): Directory to save plot to. Defaults to "test"
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Autocorrelation ' + title)
        plot_acf((self.data[1]), lags=lags, ax=axs[0], fft=True, title='Unmodified autocorrelation')
        # plt.show()
        plot_acf(np.diff(self.data[1]), lags=lags, ax=axs[1], fft=True, title='Modified autocorrelation')
        if(not os.path.isdir(f"outputs/{title}")):
            os.mkdir(f"outputs/{title}")
        fig.savefig(f'outputs/{title}/acf.png')
        plt.show()

    def calculate_autocorrelation_dfa(self, title="test"):
        """ Calculates, shows and saves Detrended Fluctuation Analysis plots with Hurst exponent for autocorrelation. 
        Depending on the value of H it is:
            * H < 0.5 - anti-correlated
            * H around 0.5 - uncorrelated, white noise
            * H > 0.5 - correlated
            * H around 1 - 1/f-noise, pink noise
            * H > 1 - non-stationary, unbounded
            * H around 1.5 - Brownian noise
        Args:
            data (_np.array_): Array with data to calculate autocorelation on.
            title (_str_, optional): Title of figure and directory name to save plots to. Defaults to "test".
        """
        a = fu.toAggregated(self.data[1])

        pydfa = fathon.DFA(a)

        winSizes = fu.linRangeByStep(5, len(a))
        revSeg = True
        polOrd = 3

        n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)

        H, H_intercept = pydfa.fitFlucVec()
        plt.title('DFA ' + title)
        plt.plot(np.log(n), np.log(F), 'ro')
        plt.plot(np.log(n), H_intercept+H*np.log(n), 'k-', label='H = {:.2f}'.format(H))
        plt.xlabel('ln(n)', fontsize=14)
        plt.ylabel('ln(F(n))', fontsize=14)
        plt.legend(loc=0, fontsize=14)

        if(not os.path.isdir(f"outputs/{title}")):
            os.mkdir(f"outputs/{title}")
        plt.savefig(f'outputs/{title}/dfa.png')
        plt.show()

    
class InteractiveIonChannel():
    def __init__(self):
        self.__a_slider = FloatSlider(min=0.0, max=10.0, step=0.1, value=1, description='a')
        self.__closed_0_slider = IntSlider(min=-10, max=10, step=1, value=-1, description='Closed value')
        self.__closed_1_slider = FloatSlider(min=0.0, max=100.0, step=0.1, value=6.0, description='Closed avg time')
        self.__opened_0_slider = IntSlider(min=-10, max=10, step=1, value=1, description='Opened value')
        self.__opened_1_slider = FloatSlider(min=0.0, max=100.0, step=0.1, value=2.0, description='Opened avg time')
        self.__D_slider = FloatSlider(min=0.1, max=100.0, step=0.1, value=2.0, description='D')
        self.__delta_t_slider = SelectionSlider(
            options=[10**-i for i in range(1, 5)],
            value=0.01,
            description='Delta t',
        )
        self.__records_slider = IntSlider(min=1000, max=100000, step=1000, value=50000, description='Records')

        self.__random_force_dropdown = Dropdown(
            options=['Gauss', 'Levy'],
            value='Gauss',
            description='Random Force',
        )
        self.__takes_previous = Checkbox(value=True, description='Takes previous values')
        self.__seed_select = IntSlider(min=0, max=99999, value=12345, step=1, description='Seed')
        self.__force_params_box = VBox()

    def __ion_channel_interactive(self, a, closed_0, closed_1, opened_0, opened_1, D, delta_t, records, random_force, takes_previous, seed, **force_params):
        """
        Generates an interactive widget for the ion channel model.
        """
        print("ion channel interactive called")
        closed = (closed_0, closed_1)
        opened = (opened_0, opened_1)
        self.ion_channel = IonChannel(a, closed, opened, D, delta_t, records, takes_prev_vals=takes_previous, seed=seed, **force_params)
        self.ion_channel.ion_channel_model(random_force)
        # return ion_channel
    
    def __update_force_params(self, *args):
        """
        Updates the force_params_box based on the selected random force.
        """
        force_type = self.__random_force_dropdown.value
        if force_type.lower() == str.lower('Levy'):
            # Define widgets specific to 'Other Force'
            alpha = FloatSlider(min=0, max=2, step=0.01, value=1.9, description='alpha')
            beta_opened = FloatSlider(min=-1.0, max=0.0, step=0.01, value=-1.0, description='beta_opened')
            beta_closed = FloatSlider(min=0.0, max=1, step=0.01, value=1.0, description='beta_closed')
            loc = IntSlider(min=0, max=100, step=1, value=0, description='location')
            scale = FloatSlider(min=0, max=100, step=0.1, value=0.1, description='scale')
            self.__force_params_box.children = [alpha, beta_closed, beta_opened, loc, scale]
        else:
            self.__force_params_box.children = []
    
    def interact(self):
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
            self.__force_params_box,
            self.__takes_previous,
            self.__random_force_dropdown
        )
        run_button = Button(description="Run Model")
        run_button.on_click(self.__on_button_click)
        display(run_button)

    def __on_button_click(self, b):
        """
        Callback for the 'Run Model' button.
        """
        print("Button Clicked")
        force_params = {child.description: child.value for child in self.__force_params_box.children}
        self.__ion_channel_interactive(
            self.__a_slider.value,
            self.__closed_0_slider.value,
            self.__closed_1_slider.value,
            self.__opened_0_slider.value,
            self.__opened_1_slider.value,
            self.__D_slider.value,
            self.__delta_t_slider.value,
            self.__records_slider.value,
            self.__random_force_dropdown.value,
            self.__takes_previous.value,
            self.__seed_select.value,
            **force_params
        )
        
        params_str = (
            f"a: {self.__a_slider.value}, "
            f"closed_0: {self.__closed_0_slider.value}, "
            f"closed_1: {self.__closed_1_slider.value}, "
            f"opened_0: {self.__opened_0_slider.value}, "
            f"opened_1: {self.__opened_1_slider.value}, "
            f"D: {self.__D_slider.value}, "
            f"delta_t: {self.__delta_t_slider.value}, "
            f"records: {self.__records_slider.value}, "
            f"random_force: {self.__random_force_dropdown.label}, "
            f"force_params: {', '.join([f'{key}: {value}' for key, value in force_params.items()])} "
            f"seed: {self.__seed_select.value}"
        )
        self.ion_channel.plot_time_series()
        self.ion_channel.plot_time_series_histogram()
        print(params_str)
        name = f"{self.__random_force_dropdown.value}_D{self.__D_slider.value}_a{self.__a_slider.value}_{self.__takes_previous.value}_{self.__seed_select.value}"
        self.ion_channel.calculate_autocorrelation_acf(title=name)
        self.ion_channel.calculate_autocorrelation_dfa(title=name)
from src.IonChannel import IonChannel
import numpy as np

from ipywidgets import FloatSlider, IntSlider, Dropdown, SelectionSlider, VBox, Button, Checkbox
from IPython.display import display
import matplotlib.pyplot as plt

import multiprocessing


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
        self.__delta_t_slider = SelectionSlider(
            options=[10**-i for i in range(3, 6)],
            value=0.0001,
            description='Delta t',
        )
        self.__a_slider = FloatSlider(min=0, max=1, step=0.001, value=0.01, description='a')
        self.__closed_0_slider = IntSlider(min=-50, max=50, step=1, value=1, description='Closed value')
        self.__closed_1_slider = IntSlider(min=1, max=10000, step=1, value=200, description='Closed avg time')
        self.__opened_0_slider = IntSlider(min=-50, max=50, step=1, value=-1, description='Opened value')
        self.__opened_1_slider = IntSlider(min=1, max=10000, step=1, value=50, description='Opened avg time')
        self.__D_slider = FloatSlider(min=0.00, max=100.00, step=0.01, value=0.1, description='D')
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
            value = 'Square',
            description = 'Model force'
        )

        self.__fft_lags = IntSlider(min=30, max=200, step=10, value=100, description='FFT lags')
        self.__takes_previous = Checkbox(value=True, description='Takes previous values')
        self.__draw_vlines_at_breakpoint = Checkbox(value=True, description='Draw breakpoints')
        self.__seed_select = IntSlider(min=0, max=99999, value=12345, step=1, description='Seed')
        self.__pol_ord_select = IntSlider(min=1, max=3, value=1, step=1, description='Polynomial degree')
        self.__prob_of_noise = FloatSlider(min=0.00, max=1.0, step=0.01, value=1.0, description='Probability of noise')
        self.__checkbox_with_inertia = Checkbox(value=True, description="Inertia")
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
            pol_ord=self.__pol_ord_select.value,
            pona=self.__prob_of_noise.value,
            **self.__force_params
        )
        self.ion_channel._generate_data(self.__force_dropdown.value, self.__random_force_dropdown.value, self.__checkbox_with_inertia.value)
    
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

        if self.__checkbox_with_inertia.value:
            gamma = FloatSlider(min=0, max=10, step=0.1, value=1, description='gamma')
            force_params.append(gamma)
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
        self.__checkbox_with_inertia.observe(self.__update_force_params, names='value')

        if self.__checkbox_with_inertia:
            gamma = FloatSlider(min=0, max=10, step=0.1, value=1, description='gamma')
            self.__force_params_box.children = [gamma]

        display(
            self.__seed_select,
            self.__a_slider,
            self.__closed_0_slider,
            self.__closed_1_slider,
            self.__opened_0_slider,
            self.__opened_1_slider,
            self.__D_slider,
            self.__prob_of_noise,
            self.__delta_t_slider,
            self.__records_slider,
            self.__checkbox_with_inertia,
            self.__takes_previous,
            self.__autocorrelation_dropdown,
            self.__fft_lags,
            self.__random_force_dropdown,
            self.__force_dropdown,
            self.__force_params_box,
            self.__draw_vlines_at_breakpoint,
            self.__pol_ord_select
        )
        run_button = Button(description="Run Model")
        run_button.on_click(self.__on_button_click)
        display(run_button)

        run_button_test = Button(description="Test MT")
        run_button_test.on_click(self.__on_button_click)
        display(run_button_test)

        run_button_test_singular = Button(description="Test Single")
        run_button_test_singular.on_click(self.__on_button_click)
        display(run_button_test_singular)
        
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
                name += f"{self.__random_force_dropdown.value}_D{D}_a{self.__a_slider.value}_{self.__seed_select.value}_n{self.__pol_ord_select.value}"
            case "Levy":
                name += f"{self.__random_force_dropdown.value}_D{D}_a{self.__a_slider.value}_alpha{self.__force_params["alpha"]}_beta{self.__force_params["beta"]}_scale{self.__force_params["scale"]}_{self.__seed_select.value}_n{self.__pol_ord_select.value}"
        

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
        dt_fig, dt_axs = self.ion_channel.plot_dwell_times_histogram()
        self.ion_channel.save_figure(dt_fig, title=name, name='dwell_times_exponent', with_subfigures=False)
        with open('outputs2/dwell_times.txt', 'w') as f:
            for item in self.ion_channel.dwell_times:
                f.write(f'{item}\n')
        f.close()


    def __multi_on_click_event(self, index):
        data = self.ion_channel.data_transposed[1]
        # self.__fig_autocorr, self.__axs_autocorr[index][0] = self.ion_channel.calculate_autocorrelation_acf(data, self.__fig_autocorr, self.__axs_autocorr[index][0], lags=self.__fft_lags.value)
        self.__fig_autocorr, self.__axs_autocorr[index] = self.ion_channel.plot_autocorrelation_dfa(data, self.__fig_autocorr, self.__axs_autocorr[index])

    @staticmethod
    def multiprocessed_worker(args):
        D, nr_of_tests, seed, force_params, force_dropdown_value, random_force_dropdown_value, delta_t_slider_value, records_slider_value, takes_previous_value, closed, opened, a_slider_value, pol_ord = args
        alpha_low = 0
        alpha_high = 0
        alpha_all = 0
        generator = np.random.Generator(np.random.PCG64(seed=seed))
        for _ in range(nr_of_tests):
            ion_channel = IonChannel(
                a=a_slider_value,
                closed=closed,
                opened=opened,
                D=D,
                delta_t=delta_t_slider_value,
                records=records_slider_value,
                takes_prev_vals=takes_previous_value,
                seed=int(generator.random()*100000),
                pol_ord=pol_ord,
                **force_params
            )
            ion_channel._generate_data(force_dropdown_value, random_force_dropdown_value, save_to_pickle=False)
            alpha_list = ion_channel.dfa(ion_channel.data_transposed[1])[2]
            alpha_low += alpha_list[0]
            alpha_high += alpha_list[1]
            alpha_all += alpha_list[2]
        return alpha_low, alpha_high, alpha_all
    
    def __on_button_click(self, b : Button):
        """
        Callback for the 'Run Model'/'Test MT'/'Test Single' button.
        """
        self.__force_params = {child.description: child.value for child in self.__force_params_box.children}
        if b.description == "Run Model":
            self.generator = np.random.Generator(np.random.PCG64(seed=self.__seed_select.value))
            self.__ion_channel_interactive()
            self.__on_click_event(self.__D_slider.value)
        if b.description == "Test MT":
            alpha_low = 0
            alpha_high = 0
            alpha_all = 0
            core_count = multiprocessing.cpu_count()
            nr_of_tests = 128
            batch_for_core = nr_of_tests // core_count
            residual_batch = nr_of_tests - (core_count * batch_for_core)
            D_list = [10.0, 50.0, 100.0, 500.0]
            for noise in ["Levy", "Gauss"]:
                self.generator = np.random.Generator(np.random.PCG64(seed=self.__seed_select.value))
                self.__random_force_dropdown.value = noise
                for D in D_list:
                    args_list = [
                        (D, batch_for_core, int(self.generator.random()*10000), self.__force_params, self.__force_dropdown.value, self.__random_force_dropdown.value, self.__delta_t_slider.value, self.__records_slider.value, self.__takes_previous.value, (self.__closed_0_slider.value, self.__closed_1_slider.value), (self.__opened_0_slider.value, self.__opened_1_slider.value), self.__a_slider.value, self.__pol_ord_select.value)
                        for _ in range(core_count)
                    ]
                    if residual_batch != 0:
                        args_list.append(
                        (D, residual_batch, int(self.generator.random()*10000), self.__force_params, self.__force_dropdown.value, self.__random_force_dropdown.value, self.__delta_t_slider.value, self.__records_slider.value, self.__takes_previous.value, (self.__closed_0_slider.value, self.__closed_1_slider.value), (self.__opened_0_slider.value, self.__opened_1_slider.value), self.__a_slider.value, self.__pol_ord_select.value)
                        )
                    with multiprocessing.Pool(core_count) as process:
                        results = process.map(self.multiprocessed_worker, args_list)
                    for result in results:
                        alpha_low += result[0]
                        alpha_high += result[1]
                        alpha_all += result[2]
                    alpha_low /= nr_of_tests
                    alpha_high /= nr_of_tests
                    alpha_all /= nr_of_tests
                    alphas = [alpha_low, alpha_high, alpha_all]
                    with open('alpha.csv', 'a') as f:
                        f.write(f'{noise}, {D}, {alphas[0]}, {alphas[1]}, {alphas[2]}\n')
        if b.description == "Test Single":
            for n in range(1, 3):
                self.__pol_ord_select.value = n
                D_list = [10.0, 50.0, 100.0, 500.0]
                for noise in ["Levy", "Gauss"]:
                    self.__fig_autocorr, self.__axs_autocorr = plt.subplots(len(D_list), 1, constrained_layout=True)
                    self.__fig_autocorr.set_size_inches(12, 6*len(D_list))
                    self.__random_force_dropdown.value = noise
                    for ind, D in enumerate(D_list):
                        self.generator = np.random.Generator(np.random.PCG64(seed=self.__seed_select.value))
                        self.__D_slider.value = D
                        self.__ion_channel_interactive()
                        self.__multi_on_click_event(ind)
                        self.__on_click_event(D)
                        self.__axs_autocorr[ind].set_title(f"DFA polynomial degree {self.__pol_ord_select.value} for {noise} D = {D}")
                    self.ion_channel.save_figure(self.__fig_autocorr, noise, name = f"DFA {noise} {self.__pol_ord_select.value}", with_subfigures=False)

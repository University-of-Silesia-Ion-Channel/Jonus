from src.IonChannel import IonChannel
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    mnoznik = 1
    a: float = 1 * mnoznik
    closed = [-1.0, 250.0]
    opened = [1.0, 50.0]
    D: float = 0.01 * mnoznik
    delta_t: float = 1.0
    records: int = 50000
    takes_prev_vals: bool = True
    seed: int = 654321
    pol_ord: int = 1
    pona: float = 1
    force_params = {'gamma' : 1.0 * mnoznik, 'alpha' : 1.9, 'beta' : 0.9, 'scale' : 1.5}
    ic = IonChannel(a, closed, opened, D, delta_t, records, takes_prev_vals, seed, pol_ord, pona, **force_params)
    ic._generate_data(model_force="Square", random_force="Gauss", with_inertia=True, save_to_pickle=False)

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    fig.set_size_inches(16, 12)
    sub_fig = ic.plot_time_series(axs[0][0], plot_breakpoints=False, plot_zoomed=True)
    ic.plot_time_series_histogram(axs[0][1])

    data = ic.data_transposed[1] # because self.data is shape (records, 3)
    fig.suptitle('Test')
    fig, axs[1][0] =  ic.calculate_autocorrelation_acf(data, fig, axs[1][0], lags=100)
    fig, axs[1][1] = ic.plot_autocorrelation_dfa(data, fig, axs[1][1])
    ic.save_figure(fig, 'test', name='test_figure', with_subfigures=False)
    ic.save_figure(sub_fig, title='test', name='zoomed_time_series', with_subfigures=False)

    point_count = 5
    print("Few first velocities:")
    print(ic.velocities[0:point_count])
    print("Current for first few points:")
    print(ic.data[0:point_count, 1:])
    vel_fig, vel_ax = plt.subplots()
    vel_fig.suptitle('Velocity changes')
    vel_ax.plot(ic.velocities)
    ic.save_figure(vel_fig, title='test', name="velocities", with_subfigures=False)

    # dt_fig, dt_axs = ic.plot_dwell_times_histogram()

    # zapisy z hormonem kontrola potencjał 50, 25 mV - wziąć wartości dla Ca = 0 i 100
    # połączyć tau z p_open
    # prnt scrn dla dr Machury
    # PSD - power spectral density - widmo mocy
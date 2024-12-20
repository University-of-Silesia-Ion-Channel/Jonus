from monostable_model import ion_channel_model, model_force_square, random_force_gauss
from ipywidgets import interact


if __name__ == '__main__':
    interact(ion_channel_model(model_force=model_force_square, random_force=random_force_gauss), D=(0, 1, 0.01))
exit()
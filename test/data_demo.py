import numpy as np
import mediapy
from tqdm import tqdm
import dataclasses

from waymax import config as _config
from waymax import dataloader
from waymax import datatypes
from waymax import visualization

config = dataclasses.replace(_config.WOD_1_1_0_TRAINING, max_num_objects=32)
data_iter = dataloader.simulator_state_generator(config=config)
scenario = next(data_iter)

img = visualization.plot_simulator_state(scenario, use_log_traj=True)
mediapy.show_image(img)

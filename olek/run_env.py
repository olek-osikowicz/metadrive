from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.examples.ppo_expert.torch_expert import torch_expert as expert

import logging
from IPython.display import Image, clear_output
import pandas as pd
from pprint import pprint

from PIL import Image
import numpy as np


def create_env(seed=0):
    # ===== Termination Scheme =====
    termination_sceme = dict(
        out_of_route_done=False,
        on_continuous_line_done=False,
        crash_vehicle_done=True,
        crash_object_done=True,
        crash_human_done=True,
    )
    # ===== Map Config =====
    map_config = {
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        BaseMap.GENERATE_CONFIG: 5,
    }

    cfg = dict(
        use_render=True, start_seed=seed, map_config=map_config, **termination_sceme
    )
    env = MetaDriveEnv(config=cfg)
    return env


if __name__ == "__main__":
    env = create_env(seed=6969)
    obs, step_info = env.reset()

    for step in range(5):

        # get action from expert driving, or a dummy action
        action = expert(env.agent, deterministic=True)
        obs, reward, tm, tr, step_info = env.step(action)

        if tm or tr:
            break

    env.close()

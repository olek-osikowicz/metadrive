from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.examples.ppo_expert.torch_expert import torch_expert as expert
from metadrive.utils import generate_gif

from metadrive.engine.logger import get_logger
from pathlib import Path

import json

from PIL import Image
import numpy as np
import logging
import time
import datetime

logger = get_logger()
now = datetime.datetime.now()
file_handler = logging.FileHandler(f"logs/{now.strftime('%Y-%m-%d_%H:%M')}.log")
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def max_touching_distance(ego, npc):
    ego_length, ego_width = ego.get_state()["length"], ego.get_state()["width"]
    npc_length, npc_width = npc.get_state()["length"], npc.get_state()["width"]

    # Pythagorean theorem
    dist = np.sqrt(
        (ego_length / 2 + npc_length / 2) ** 2 + (ego_width / 2 + npc_width / 2) ** 2
    )
    # add 10% just in case
    dist = dist * 1.1
    return dist


def get_crashed_vehicles(env) -> set:

    ret = set()
    ego = env.agent
    npcs = env.agent_manager.get_objects()

    # iterate over npc vehicles
    for npc in npcs.values():
        npc_state = npc.get_state()

        # if npc crashed
        if npc_state["crash_vehicle"]:

            # calculate distance beetween them
            distance = np.linalg.norm(ego.position - npc.position)

            # calculate max_touching_distance, (collision threshold)
            max_dist = max_touching_distance(ego, npc)

            # pprint(f"{distance = }")
            # pprint(f"{max_touching_distance(ego, npc) = }")

            if npc.id is not ego.id and distance < max_dist:
                ret.add(npc.id)

    return ret


def serialize_step_info(info) -> dict:
    """Convert numpy floats to native so can be serialized."""
    info["action"] = [float(x) for x in info["action"]]
    info["raw_action"] = [float(x) for x in info["raw_action"]]
    return info


def process_timestamps(start_ts, initialized_ts, scenario_done_ts, env_closed_ts):
    """Calculate and log time it took to initialise and run the env."""

    init_time = initialized_ts - start_ts
    logger.info(f"Initializing the env took {init_time:.2f}s")

    scenario_time = scenario_done_ts - initialized_ts
    logger.info(f"Running the scenario took {scenario_time:.2f}s")

    closing_time = env_closed_ts - scenario_done_ts
    logger.info(f"Closing the env took {closing_time:.2f}s")

    total_time = env_closed_ts - start_ts
    logger.info(f"Total scenario execution took {total_time:.2f}s")

    return locals()


def get_map_img(env):
    """Get map image of the current environment"""
    map = env.current_map.get_semantic_map(
        env.current_map.get_center_point(),
    )
    map = map.squeeze()  # reduce dimensionality
    map = (map * 255 * 4).astype(np.uint8)
    img = Image.fromarray(map)
    return img


def get_bv_state(env) -> list:
    def filter_vehicle_state(v_state: dict) -> dict:
        wanted_keys = ["length", "width", "height", "spawn_road", "destination"]
        return {key: v_state[key] for key in wanted_keys}

    vehicles = env.agent_manager.get_objects()
    bvs_states = [filter_vehicle_state(v.get_state()) for v in vehicles.values()]
    return bvs_states


class ScenarioRunner:

    def __init__(
        self,
        save_dir: str,
        seed: int = 0,
        decision_repeat: int = 5,
        dt: float = 0.02,
        traffic_density: float = 0.1,
    ) -> None:

        self.seed = seed
        self.decision_repeat = decision_repeat
        self.dt = dt
        self.traffic_density = traffic_density
        save_dir = Path(save_dir)
        self.save_path = save_dir / f"dr_{decision_repeat}_dt_{dt}"
        self.save_path.mkdir(parents=True, exist_ok=True)
        assert self.save_path.exists(), f"{self.save_path} does not exist!"
        self.crashed_vehicles = set()

    def get_max_steps(self, env: MetaDriveEnv):
        """
        Return maximum number of simulation steps.

        Assume minimal target velocity e.g. 2m/s.
        Dependant on the total route length.
        Adaptable to fidelity parameters.
        """

        distance = env.agent.navigation.total_length
        V_min = 2.0  # [m/s]  # set minimal velocity to 2m/s

        max_steps = distance / (V_min * self.decision_repeat * self.dt)
        logger.info(f"Calculating max steps with: ")
        logger.info(
            f"{V_min = }, {self.decision_repeat = }, {self.dt = }, {distance = :.2f}, {round(max_steps) = }"
        )
        return round(max_steps)

    def state_action_loop(
        self, env: MetaDriveEnv, max_steps: int, record_gif: bool = False
    ) -> list:
        """Runs the simulations steps until max_steps limit hit"""
        logger.info(f"Launching the scenario with {record_gif = }")
        steps_infos = []
        frames = []
        while True:

            action = expert(env.agent, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if info["episode_length"] == max_steps:
                truncated = True
                info["max_step"] = True
                logger.info("Time out reached!")

            if record_gif:
                frames.append(env.render(mode="topdown", window=False))

            if info["crash_vehicle"]:
                self.crashed_vehicles.update(get_crashed_vehicles(env))

            steps_infos.append(serialize_step_info(info))

            if terminated or truncated:
                break

        if record_gif:
            generate_gif(frames, gif_name=f"{self.save_path}/{self.seed}.gif")

        return steps_infos

    def create_env(self) -> MetaDriveEnv:
        # ===== Fidelity Config =====
        fidelity_params = dict(
            decision_repeat=self.decision_repeat, physics_world_step_size=self.dt
        )

        # ===== Termination Scheme =====
        termination_sceme = dict(
            out_of_route_done=False,
            on_continuous_line_done=False,
            crash_vehicle_done=False,
            crash_object_done=False,
            crash_human_done=False,
        )
        # ===== Map Config =====
        map_config = {
            BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
            BaseMap.GENERATE_CONFIG: 5,  # 20 block
        }

        cfg = dict(
            # use_render=True,
            log_level=logging.INFO,  # logging.DEBUG
            start_seed=self.seed,
            traffic_density=self.traffic_density,
            map_config=map_config,
            **termination_sceme,
            **fidelity_params,
        )
        env = MetaDriveEnv(config=cfg)
        return env

    def data_exists(self) -> bool:
        lst = list(self.save_path.glob(f"{self.seed}.json"))
        return bool(lst)

    def run_scenario(
        self, record_gif=False, repeat=False, dry_run=False, save_map=False
    ):

        start_ts = time.perf_counter()

        if self.data_exists() and not repeat:
            logger.info("Data for this scenario exists skipping.")
            return

        # initialize
        env = self.create_env()
        _, reset_info = env.reset()

        scenario_data = {}
        scenario_data["def.map_seq"] = env.current_map.get_meta_data()["block_sequence"]
        scenario_data["def.bv_data"] = get_bv_state(env)
        scenario_data["def.spawn_lane_index"] = env.agent.config["spawn_lane_index"][-1]
        max_step = self.get_max_steps(env)
        scenario_data["def.max_steps"] = max_step

        initialized_ts = time.perf_counter()

        # running loop if it's not a dry run
        steps_info = []
        if not dry_run:
            steps_info = self.state_action_loop(env, max_step, record_gif)

        scenario_done_ts = time.perf_counter()

        env.close()

        env_closed_ts = time.perf_counter()
        # save execution metadata
        scenario_data.update(
            process_timestamps(
                start_ts, initialized_ts, scenario_done_ts, env_closed_ts
            )
        )

        steps_info.insert(0, reset_info)
        scenario_data["steps_infos"] = steps_info
        scenario_data["n_crashed_vehicles"] = len(self.crashed_vehicles)

        with open(self.save_path / f"{self.seed}.json", "w") as f:
            json.dump(scenario_data, f, indent=4)

        if save_map:
            get_map_img(env).save(self.save_path / f"{self.seed}.png")

        data_saved_ts = time.perf_counter()
        logger.info(f"Saving data took {data_saved_ts-env_closed_ts:.2f}s")
        logger.info(f"Running scenario finished.")


if __name__ == "__main__":

    # ScenarioRunner(seed=123, decision_repeat=6, dt=0.03).run_scenario()
    ScenarioRunner("test", seed=123, decision_repeat=5, dt=0.02).run_scenario(
        repeat=True, record_gif=True
    )

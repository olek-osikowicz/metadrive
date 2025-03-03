from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.examples.ppo_expert.torch_expert import torch_expert as expert
from metadrive.utils import generate_gif

from metadrive.engine.logger import get_logger
from pathlib import Path

import json

from PIL import Image
import cv2
import numpy as np
import logging
import time
import datetime

logger = get_logger()
now = datetime.datetime.now()
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"{now.strftime('%Y-%m-%d_%H:%M')}.log"
file_handler = logging.FileHandler(log_path)
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


def process_timestamps(start_ts, initialized_ts, scenario_done_ts, env_closed_ts) -> dict:
    """
    Calculate and log the time it took to initialize and run the environment.

    Returns:
        dict: A dictionary with the time data.
    """

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


def get_bv_state(env) -> dict:
    vehicles = env.agent_manager.get_objects()
    # Get state of each vehicle except the agent
    return {k: v.get_state() for k, v in vehicles.items() if k != env.agent.id}


def scenario_file_exists(file_path: Path) -> bool:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json.load(file)  # Try to parse JSON
        return True  # No error means it's valid JSON
    except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
        return False


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
        self.fps = round(1 / (dt * decision_repeat))
        self.traffic_density = traffic_density
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.save_dir / f"{self.fps}_{self.seed}.json"

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
        max_time = distance / V_min  # [s] maximum time allowed to reach the destination
        max_steps = round(max_time * self.fps)  # maximum number of simulation steps frames

        logger.info(f"Calculating max steps with: ")
        logger.info(
            f"{V_min = }, {distance = }, {max_time = }, {self.fps = } {max_steps = }"
        )

        return max_steps

    def get_video_writer(self) -> cv2.VideoWriter:
        output_filename = self.file_path.with_suffix(".mp4")
        logger.info(f"Saving render to {output_filename}")
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        frame_size = (800, 800)
        return cv2.VideoWriter(output_filename, codec, self.fps, frame_size)

    def state_action_loop(
        self, env: MetaDriveEnv, max_steps: int, record: bool = False
    ) -> list:
        """Runs the simulations steps until max_steps limit hit"""
        logger.info(f"Launching the scenario with {record = }")
        steps_infos = []
        if record:
            video_writer = self.get_video_writer()

        while True:

            action = expert(env.agent, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            info["bv_data"] = get_bv_state(env)

            if info["episode_length"] == max_steps:
                truncated = True
                info["max_step"] = True
                logger.info("Time out reached!")

            if record:
                frame = env.render(mode="topdown", window=False)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            if info["crash_vehicle"]:
                self.crashed_vehicles.update(get_crashed_vehicles(env))

            steps_infos.append(serialize_step_info(info))

            if terminated or truncated:
                break

        if record:
            video_writer.release()

        return steps_infos

    def get_config(self) -> dict:
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
        return cfg

    def get_scenario_definition_from_env(self, env: MetaDriveEnv) -> dict:
        """Get data from the environment"""
        data = {}
        data["fid.dt"] = self.dt
        data["fid.decision_repeat"] = self.decision_repeat
        data["fid.fps"] = self.fps
        data["def.seed"] = self.seed
        data["def.map_seq"] = env.current_map.get_meta_data()["block_sequence"]
        data["def.bv_data"] = get_bv_state(env)
        data["def.spawn_lane_index"] = env.agent.config["spawn_lane_index"][-1]
        data["def.distance"] = env.agent.navigation.total_length
        max_step = self.get_max_steps(env)
        data["def.max_steps"] = max_step

        return data

    def run_scenario(
        self, record=False, repeat=False, dry_run=False, save_map=False
    ) -> dict:
        """
        Run a scenario and save the results.

        Parameters:
        - record (bool): If True, records a video of the scenario. Default is False.
        - repeat (bool): If True, runs the scenario even if data already exists. Default is False.
        - dry_run (bool): If True, runs the scenario without executing the main loop. Default is False.
        - save_map (bool): If True, saves the map image. Default is False.

        Returns:
        - dict: A dictionary containing timing information of different stages of the scenario execution.

        The function performs the following steps:
        1. Checks if data for the scenario already exists and skips execution if `repeat` is False.
        2. Initializes the environment and resets it.
        3. Collects initial scenario data including map sequence, vehicle state, spawn lane index, and maximum steps.
        4. Runs the main loop to collect step information if `dry_run` is False.
        5. Closes the environment and processes timestamps for different stages.
        6. Saves the scenario data and step information to a JSON file.
        7. Optionally saves the map image if `save_map` is True.
        8. Logs the time taken to save data and indicates the completion of the scenario run.
        """

        start_ts = time.perf_counter()

        if scenario_file_exists(self.file_path) and not repeat:
            logger.info(f"Data for scenario {self.file_path} exists skipping.")
            return

        # initialize
        env = MetaDriveEnv(config=self.get_config())
        _, reset_info = env.reset()

        scenario_data = {**self.get_scenario_definition_from_env(env)}
        if save_map:
            get_map_img(env).save(self.file_path.with_suffix(".png"))

        initialized_ts = time.perf_counter()

        # running loop if it's not a dry run
        steps_info = []
        if not dry_run:
            steps_info = self.state_action_loop(env, scenario_data["def.max_steps"], record)

        scenario_done_ts = time.perf_counter()

        env.close()

        env_closed_ts = time.perf_counter()
        # save execution metadata
        timings = process_timestamps(
            start_ts, initialized_ts, scenario_done_ts, env_closed_ts
        )
        scenario_data.update(timings)

        steps_info.insert(0, reset_info)
        scenario_data["eval.steps_infos"] = steps_info
        scenario_data["eval.n_crashed_vehicles"] = len(self.crashed_vehicles)

        with open(self.file_path, "w") as f:
            json.dump(scenario_data, f, indent=4)

        data_saved_ts = time.perf_counter()
        logger.info(f"Saving data took {data_saved_ts-env_closed_ts:.2f}s")
        logger.info(f"Running scenario finished.")
        return timings


if __name__ == "__main__":

    # ScenarioRunner(seed=123, decision_repeat=6, dt=0.03).run_scenario()
    ScenarioRunner("test", seed=123, decision_repeat=5, dt=0.02).run_scenario(
        repeat=True, record=True
    )

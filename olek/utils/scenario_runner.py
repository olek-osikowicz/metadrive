from functools import cached_property
from itertools import count
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.engine.logger import get_logger
from metadrive.examples.ppo_expert.numpy_expert import expert

# from metadrive.examples.ppo_expert.torch_expert import torch_expert as expert

from pathlib import Path
import json
import numpy as np
import logging
import time
import cv2

WORLD_FPS = 60
RECORD_VIDEO_FPS = 10

# pixels per meter for recording
RENDER_SCALING = 3


class MetaDriveJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, type):
            return obj.__name__
        return super().default(obj)


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
        save_dir: str | Path,
        seed: int = 0,
        ads_fps: int = 10,
    ) -> None:

        start_ts = time.perf_counter()

        self.log = get_logger()

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log.info(f"Saving data to {self.save_dir}")
        self.file_path = self.save_dir / f"{ads_fps}_{seed}.json"

        self.seed = seed
        assert WORLD_FPS % ads_fps == 0, "ADS FPS must be a divisor of worlds FPS"
        self.ads_fps = ads_fps
        self.crashed_vehicles = set()

        # initialize
        self.env = MetaDriveEnv(config=self.get_config())
        _, reset_info = self.env.reset()

        assert self.env.config["decision_repeat"] == 1, "Decision repeat must be 1"

        self.scenario_data = self.initialize_scenario_data()
        self.scenario_data["reset_info"] = reset_info

        self.timings = {"init_time": start_ts - time.perf_counter(), "agent_time": 0.0}

    def __del__(self):
        self.env.close()

    def max_touching_distance(self, ego, npc):
        ego_length, ego_width = ego.get_state()["length"], ego.get_state()["width"]
        npc_length, npc_width = npc.get_state()["length"], npc.get_state()["width"]

        # Pythagorean theorem
        dist = np.sqrt(
            (ego_length / 2 + npc_length / 2) ** 2 + (ego_width / 2 + npc_width / 2) ** 2
        )
        # add 10% just in case
        dist = dist * 1.1
        return dist

    def get_crashed_vehicles(self) -> set:

        ret = set()
        ego = self.env.agent
        npcs = self.env.agent_manager.get_objects()

        # iterate over npc vehicles
        for npc in npcs.values():
            npc_state = npc.get_state()

            # if npc crashed
            if npc_state["crash_vehicle"]:

                # calculate distance beetween them
                distance = np.linalg.norm(ego.position - npc.position)

                # calculate max_touching_distance, (collision threshold)
                max_dist = self.max_touching_distance(ego, npc)

                # pprint(f"{distance = }")
                # pprint(f"{max_touching_distance(ego, npc) = }")

                if npc.id is not ego.id and distance < max_dist:
                    ret.add(npc.id)

        return ret

    def get_bv_state(self) -> dict:
        vehicles = self.env.agent_manager.get_objects()
        # Get state of each vehicle except the agent
        return {k: v.get_state() for k, v in vehicles.items() if k != self.env.agent.id}

    @cached_property
    def max_steps(self) -> int:
        """
        Return maximum number of simulation steps.

        Assume minimal target velocity e.g. 2m/s.
        Dependant on the total route length.
        Adaptable to fidelity parameters.
        """

        distance = self.env.agent.navigation.total_length
        V_min = 2.0  # [m/s]  # set minimal velocity to 2m/s
        max_time = distance / V_min  # [s] maximum time allowed to reach the destination
        max_steps = round(max_time * WORLD_FPS)  # maximum number of simulation steps frames

        self.log.info(f"Calculating max steps with: ")
        self.log.info(
            f"{V_min = :.2f}, {distance = :.2f}, {max_time = :.2f}, {WORLD_FPS = } {max_steps = }"
        )

        return max_steps

    def get_config(self, dr=1) -> dict:

        dt = 1 / WORLD_FPS
        # ===== Termination Scheme =====
        termination_scheme = dict(
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

        return dict(
            # use_render=True,
            log_level=logging.INFO,  # logging.DEBUG
            traffic_density=0.01,
            traffic_mode="respawn",
            random_traffic=False,
            map_config=map_config,
            **termination_scheme,
            decision_repeat=dr,
            physics_world_step_size=dt,
            start_seed=self.seed,
        )

    def initialize_scenario_data(self) -> dict:
        """Get data from the environment"""
        data = {}

        data["fid.ads_fps"] = self.ads_fps
        data["fid.world_fps"] = WORLD_FPS
        data["def.seed"] = self.seed
        data["def.map_seq"] = self.env.current_map.get_meta_data()["block_sequence"]
        data["def.bv_data"] = self.get_bv_state()
        data["def.spawn_lane_index"] = self.env.agent.config["spawn_lane_index"][-1]
        data["def.distance"] = self.env.agent.navigation.total_length
        data["def.max_steps"] = self.max_steps
        data["def.env_config"] = self.env.config.get_serializable_dict()

        return data

    def run_scenario(self, record=False, repeat=False, dry_run=False) -> dict:
        """
        Run a scenario and save the results.

        Parameters:
        - record (bool): If True, records a video of the scenario. Default is False.
        - repeat (bool): If True, runs the scenario even if data already exists. Default is False.
        - dry_run (bool): If True, runs the scenario without executing the main loop. Default is False.

        Returns:
        - dict: A dictionary containing timing information of different stages of the scenario execution.
        """
        if scenario_file_exists(self.file_path) and not repeat:
            self.log.info(f"Data for scenario {self.file_path} exists skipping.")
            return

        scenario_start = time.perf_counter()
        # running loop if it's not a dry run
        if not dry_run:
            steps_infos = self.state_action_loop(record)
        else:
            steps_infos = []

        self.timings["scenario_time"] = time.perf_counter() - scenario_start
        self.log.info(f"Running scenario finished.")

        # CLOSE THE ENV
        cleanup_start = time.perf_counter()
        self.env.close()
        self.timings["closing_time"] = time.perf_counter() - cleanup_start

        # save execution metadata
        self.scenario_data.update(self.timings)

        self.scenario_data["eval.steps_infos"] = steps_infos
        self.scenario_data["eval.n_crashed_vehicles"] = len(self.crashed_vehicles)

        with open(self.file_path, "w") as f:
            json.dump(self.scenario_data, f, indent=4, cls=MetaDriveJSONEncoder)

        self.log.info(f"Data saved!")

    def state_action_loop(self, record: bool = False) -> list:
        """Runs the simulations steps until max_steps limit hit"""
        self.log.info(f"Launching the scenario with {record = }")
        steps_infos = []
        if record:
            video_writer = self.get_video_writer()
        skip_rate = WORLD_FPS // self.ads_fps
        self.log.info(f"World FPS: {WORLD_FPS}, ADS FPS: {self.ads_fps}, Ratio: {skip_rate}")

        for step_no in count():
            self.log.debug(f"Step {step_no}")
            if step_no % skip_rate == 0:
                self.log.debug(f"Getting agent's action")
                agent_start = time.perf_counter()
                action = expert(self.env.agent, deterministic=True)
                self.timings["agent_time"] += time.perf_counter() - agent_start

            obs, reward, terminated, truncated, info = self.env.step(action)

            # info["bv_data"] = self.get_bv_state()

            if info["episode_length"] == self.max_steps:
                truncated = True
                info["max_step"] = True
                self.log.info("Time out reached!")

            if record and step_no % (WORLD_FPS // RECORD_VIDEO_FPS) == 0:
                frame = self.get_frame()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)

            if info["crash_vehicle"]:
                self.crashed_vehicles.update(self.get_crashed_vehicles())

            steps_infos.append(info)

            if terminated or truncated:
                break

        if record:
            video_writer.release()

        return steps_infos

    def get_video_writer(self) -> cv2.VideoWriter:
        output_filename = self.file_path.with_suffix(".mp4")
        self.log.info(f"Saving render to {output_filename}")
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        frame_size = self.screen_size
        return cv2.VideoWriter(output_filename, codec, RECORD_VIDEO_FPS, frame_size)

    @cached_property
    def screen_size(self) -> tuple:
        b_box = self.env.current_map.road_network.get_bounding_box()
        x_len, y_len = b_box[1] - b_box[0], b_box[3] - b_box[2]
        width = int(x_len * RENDER_SCALING * 1.05)
        height = int(y_len * RENDER_SCALING * 1.05)
        return width, height

    @cached_property
    def center_point(self) -> np.ndarray:
        return self.env.current_map.get_center_point()

    def get_frame(self):

        return self.env.render(
            mode="topdown",
            window=False,
            screen_size=self.screen_size,
            camera_position=self.center_point,
            scaling=RENDER_SCALING,
            draw_contour=True,
            draw_target_vehicle_trajectory=True,
            semantic_map=False,
            num_stack=1,
        )


if __name__ == "__main__":
    pass

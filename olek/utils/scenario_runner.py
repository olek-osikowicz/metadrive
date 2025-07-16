import json
import logging
import time
from functools import cached_property
from itertools import count
from pathlib import Path

import cv2
import numpy as np

from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.engine.logger import get_logger
from metadrive.envs.metadrive_env import MetaDriveEnv

# from metadrive.examples.ppo_expert.numpy_expert import expert
from metadrive.examples.ppo_expert.torch_expert import torch_expert as expert

log = get_logger()
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
    except (json.JSONDecodeError, FileNotFoundError, IOError):
        return False


class ScenarioRunner:
    def __init__(
        self,
        save_dir: str | Path,
        seed: int = 0,
        ads_fps: int = 10,
        config: dict = {},
    ) -> None:
        start_ts = time.perf_counter()

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving data to {self.save_dir}")
        self.file_path = self.save_dir / f"{ads_fps}_{seed}.json"

        self.seed = int(seed)
        assert WORLD_FPS % ads_fps == 0, "ADS FPS must be a divisor of worlds FPS"
        self.ads_fps = ads_fps
        self.crashed_vehicles = set()

        # initialize driving enviroment
        env_config = self.get_default_config()
        env_config.update(config)
        self.env = MetaDriveEnv(config=env_config)
        _, reset_info = self.env.reset()
        log.setLevel(logging.INFO)

        assert self.env.config["decision_repeat"] == 1, "Decision repeat must be 1"

        self.scenario_data = self.initialize_scenario_data()
        self.scenario_data["def.reset_info"] = reset_info

        self.timings = {"init_time": time.perf_counter() - start_ts, "agent_time": 0.0}

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

    def get_vehicles_state(self) -> dict:
        vehicles = self.env.agent_manager.get_objects()
        # Get state of each vehicle
        return {k: v.get_state() for k, v in vehicles.items()}

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

        log.debug(
            f"Calculating max steps with: {V_min = :.2f}, {distance = :.2f}, {max_time = :.2f}, {WORLD_FPS = } {max_steps = }"
        )

        return max_steps

    def get_default_config(self, dr=1) -> dict:
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
            log_level=logging.DEBUG,  # logging.INFO # logging.DEBUG
            traffic_density=0.01,
            # traffic_mode="respawn",
            traffic_mode="basic",
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
        data["def.vehicles_data"] = self.get_vehicles_state()
        data["def.spawn_lane_index"] = self.env.agent.config["spawn_lane_index"][-1]
        data["def.distance"] = self.env.agent.navigation.total_length
        data["def.max_steps"] = self.max_steps
        data["def.env_config"] = self.env.config.get_serializable_dict()

        return data

    def run_scenario(self, record=False, repeat=False, dry_run=False):
        """
        Run a scenario and save the results.

        Parameters:
        - record (bool): If True, records a video of the scenario. Default is False.
        - repeat (bool): If True, runs the scenario even if data already exists. Default is False.
        - dry_run (bool): If True, runs the scenario without executing the main loop. Default is False.
        """
        if scenario_file_exists(self.file_path) and not repeat:
            log.info(f"Data for scenario {self.file_path} exists skipping.")
            return

        scenario_start = time.perf_counter()
        # running loop if it's not a dry run
        if not dry_run:
            steps_infos = self.state_action_loop(record)
        else:
            steps_infos = []
            if record:
                frame = self.get_bev_frame()
                cv2.imwrite(str(self.file_path.with_suffix(".png")), frame)

        self.timings["scenario_time"] = time.perf_counter() - scenario_start
        log.info("Running scenario finished.")

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

        log.info("Data saved!")

    def init_recording(self):
        log.info("Recording semantic BEV video...")
        self.bev_video_writer = self.get_video_writer(
            self.screen_size, self.file_path.with_stem(self.file_path.stem + "_bev")
        )
        if self.env.engine.win:
            log.info("Recording TPV video...")
            self.tpv_video_writer = self.get_video_writer(
                (1200, 900), self.file_path.with_stem(self.file_path.stem + "_tpv")
            )

    def tick_recording(self):
        self.bev_video_writer.write(self.get_bev_frame())
        if self.env.engine.win:
            self.tpv_video_writer.write(self.get_tpv_frame())

    def save_recording(self):
        self.bev_video_writer.release()
        if self.env.engine.win:
            self.tpv_video_writer.release()

    def state_action_loop(self, record: bool = False) -> list:
        """Runs the simulations steps until max_steps limit hit"""
        log.info(f"Launching the scenario with {record = }")
        steps_infos = []
        skip_rate = WORLD_FPS // self.ads_fps
        log.info(f"World FPS: {WORLD_FPS}, ADS FPS: {self.ads_fps}, Ratio: {skip_rate}")

        if record:
            self.init_recording()

        for step_no in count():
            log.debug(f"Step {step_no}")
            if step_no % skip_rate == 0:
                log.debug("Getting agent's action")
                agent_start = time.perf_counter()
                action = expert(self.env.agent, deterministic=True)
                self.timings["agent_time"] += time.perf_counter() - agent_start

            obs, reward, terminated, truncated, info = self.env.step(action)

            # info["vehicles_data"] = self.get_bv_state()

            if info["episode_length"] == self.max_steps:
                truncated = True
                info["max_step"] = True
                log.info("Time out reached!")

            if record and step_no % (WORLD_FPS // RECORD_VIDEO_FPS) == 0:
                self.tick_recording()

            if info["crash_vehicle"]:
                self.crashed_vehicles.update(self.get_crashed_vehicles())

            steps_infos.append(info)

            if terminated or truncated:
                break

        if record:
            self.save_recording()

        return steps_infos

    def get_video_writer(self, frame_size: tuple[int, int], path: Path) -> cv2.VideoWriter:
        output_filename = path.with_suffix(".mp4")
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving render to {output_filename}")
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(output_filename, codec, RECORD_VIDEO_FPS, frame_size)

    @cached_property
    def screen_size(self) -> tuple:
        b_box = self.env.current_map.road_network.get_bounding_box()
        x_len, y_len = b_box[1] - b_box[0], b_box[3] - b_box[2]
        width = int(x_len * RENDER_SCALING * 1.05)
        height = int(y_len * RENDER_SCALING * 1.05)
        return width, height

    def get_tpv_frame(self):
        origin_img = self.env.engine.win.getDisplayRegion(1).getScreenshot()
        frame = np.frombuffer(origin_img.getRamImage().getData(), dtype=np.uint8)

        frame = frame.reshape((origin_img.getYSize(), origin_img.getXSize(), 4))
        frame = frame[::-1]
        frame = frame[..., :3]

        return frame

    def get_bev_frame(self):
        frame = self.env.render(
            mode="topdown",
            window=False,
            screen_size=self.screen_size,
            camera_position=self.env.current_map.get_center_point(),
            scaling=RENDER_SCALING,
            draw_contour=True,
            draw_target_vehicle_trajectory=True,
            semantic_map=False,
            num_stack=1,
        )
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame

    def get_evaluation_cost(self) -> int:
        """Return a cost of running a scenario"""
        # Currently implemented simply
        return self.ads_fps // 10


if __name__ == "__main__":
    pass

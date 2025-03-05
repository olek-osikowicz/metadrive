from metadrive.envs.metadrive_env import MetaDriveEnv
from utils.scenario_runner import logger, ScenarioRunner
from tqdm import tqdm
import json
import logging

SAVE_DIR = (
    "/home/olek/Documents/dev/metadrive-multifidelity-data/data/fast_sampled_scenarios"
)


class ScenarioSampler(ScenarioRunner):

    def sample_scenario(self):

        file_path = self.save_path / f"{self.seed}.json"
        if file_path.exists():
            logger.info(f"Scenario {self.seed} already exists, skipping...")
            return
        # Initialize metadrive environment
        cfg = self.get_config()
        cfg["log_level"] = logging.CRITICAL

        env = MetaDriveEnv(config=cfg)
        env.reset()

        scenario_data = self.get_scenario_definition_from_env(env)
        with open(file_path, "w") as f:
            json.dump(scenario_data, f, indent=4)

        env.close()


if __name__ == "__main__":
    pass

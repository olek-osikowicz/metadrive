from pathlib import Path
from utils.scenario_runner import ScenarioRunner
from tqdm import tqdm
import itertools


def get_dt(fps):
    # Assume default value
    DECISION_REPEAT = 5
    interval = 1 / fps
    dt = interval / DECISION_REPEAT
    return dt


if __name__ == "__main__":

    SAVE_DIR = Path("/media/olek/2TB_HDD/metadrive-data/playground/high_fid_test")
    SEED_RANGE = range(0, 1000)
    REPS = 10
    FPS_RANGE = [10, 20, 30, 40, 50, 60]

    for seed in SEED_RANGE:
        for rep in range(REPS):
            save_path = SAVE_DIR / f"{rep}"
            for fps in FPS_RANGE:
                dt = get_dt(fps)
                ScenarioRunner(save_path, seed, dt=dt, traffic_density=0.0).run_scenario(
                    record=True,
                    repeat=True,
                )

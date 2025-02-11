from pathlib import Path
from utils.scenario_runner import ScenarioRunner
from tqdm import tqdm
import itertools
from tqdm.contrib.concurrent import process_map
from tqdm.contrib.logging import logging_redirect_tqdm
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

HDD_DIR = Path("/media/olek/2TB_HDD/metadrive-data/playground")


def get_dt(fps):
    # Assume default value
    DECISION_REPEAT = 5
    interval = 1 / fps
    dt = interval / DECISION_REPEAT
    return dt


def run_scenario(args):

    seed, use_cars, rep, fps = args
    traffic_density = 0.1 if use_cars else 0.0
    save_dir = HDD_DIR / "withcars" if use_cars else HDD_DIR / "nocars"

    save_dir = save_dir / f"{rep}"
    dt = get_dt(fps)
    ScenarioRunner(save_dir, seed, dt=dt, traffic_density=traffic_density).run_scenario(
        record=True
    )


if __name__ == "__main__":

    SEED_RANGE = range(1000)
    REPS = 10
    FPS_RANGE = [60, 50, 40, 30, 20, 10]
    use_cars = [True, False]
    jobs = list(itertools.product(SEED_RANGE, use_cars, range(REPS), FPS_RANGE))
    print(jobs)
    with logging_redirect_tqdm():
        process_map(run_scenario, jobs, max_workers=16)

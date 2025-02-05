from utils.scenario_runner import ScenarioRunner, logger
from utils.scenario_sampler import ScenarioSampler
import time
import random
from tqdm.contrib.concurrent import process_map  # or thread_map

import multiprocessing
from multiprocessing import Pool

multiprocessing.set_start_method("spawn", force=True)

SAVE_DIR = "/home/ubuntu/mnt"


def run_scenario(seed):
    logger.info(f"Running scenario {seed}")
    ScenarioRunner(SAVE_DIR, seed, traffic_density=0.0).run_scenario()


if __name__ == "__main__":

    start_ts = time.time()
    logger.info("Running validation scenarios...")
    N_SAMPLES = 100_000
    START_SEED = 10**6

    seed_range = [START_SEED + i for i in range(N_SAMPLES)]
    with Pool(maxtasksperchild=1) as p:
        p.map(run_scenario, seed_range)

    logger.info("All scenarios finished!")

    logger.info(f"Running {N_SAMPLES} scenarios took {time.time()-start_ts:.2f}s")

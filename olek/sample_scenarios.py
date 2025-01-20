from utils.scenario_runner import ScenarioRunner, logger
from utils.scenario_sampler import ScenarioSampler
import time
import random
from tqdm.contrib.concurrent import process_map  # or thread_map

import multiprocessing
from multiprocessing import Pool

multiprocessing.set_start_method("spawn", force=True)

SAVE_DIR = (
    "/home/olek/Documents/dev/metadrive-multifidelity-data/data/fast_sampled_scenarios"
)


def sample_scenario(seed):
    ScenarioSampler(SAVE_DIR, seed, traffic_density=0.0).sample_scenario()


if __name__ == "__main__":

    start_ts = time.time()
    logger.info("Sampling scenarios...")
    N_SAMPLES = 100_000
    START_SEED = 10**6

    seed_range = [START_SEED + i for i in range(N_SAMPLES)]
    process_map(sample_scenario, seed_range, chunksize=100, max_workers=12)

    logger.info("Scenarios sampled!")

    logger.info(f"Sampling {N_SAMPLES} scenarios took {time.time()-start_ts:.2f}s")

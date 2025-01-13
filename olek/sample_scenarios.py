from scenario_runner import ScenarioRunner, logger
import time
import random
import multiprocessing
from multiprocessing import Pool

multiprocessing.set_start_method("spawn", force=True)

SAVE_DIR = (
    "/home/olek/Documents/dev/metadrive-multifidelity-data/data/sampled_scenarios"
)


def sample_scenario(seed):
    DR, DT = 5, 0.02
    ScenarioRunner(SAVE_DIR, seed, DR, DT, traffic_density=0.0).run_scenario(
        dry_run=True,
    )


if __name__ == "__main__":

    start_ts = time.time()
    random.seed(2137)
    N_SAMPLES = 10_000
    env_seeds = [random.randint(2**30, 2**31) for _ in range(N_SAMPLES)]
    with Pool() as p:
        p.map(sample_scenario, env_seeds)

    logger.info(f"Sampling {N_SAMPLES} scenarios took {time.time()-start_ts:.2f}s")

from scenario_runner import ScenarioRunner, logger

from itertools import count, product
import random
import time
import multiprocessing
from multiprocessing import Pool

multiprocessing.set_start_method("spawn", force=True)

SAVE_DIR = "/home/olek/Documents/dev/metadrive-multifidelity-data/data/randomsearch_2"


def do_random_search(rep, dr, dt, time_budget=60):
    logger.info(f"Starting random search for: {rep = }")
    start_ts = time.time()

    random.seed(rep)
    for it in count():
        logger.info(f"Starting iteration {it = }")
        env_seed = random.randint(100, 10**10)
        path = f"{SAVE_DIR}/{rep}/{it}"
        ScenarioRunner(path, env_seed, dr, dt, traffic_density=0).run_scenario(
            record_gif=False, repeat=True
        )

        if time.time() - start_ts > time_budget:
            logger.info(f"Time elapsed!")
            break

    logger.info(f"Random search finished for: {rep = }!")


if __name__ == "__main__":

    TIME_BUDGET = 2 * 60  # 2 mins
    N_REPETITIONS = 50
    DR, DT = 5, 0.02
    N_PROCESSES = 10

    logger.info(f"Starting multiprocessing random search! with {N_PROCESSES = }")

    rs_params = [(rep, DR, DT, TIME_BUDGET) for rep in range(N_REPETITIONS)]

    with Pool(N_PROCESSES, maxtasksperchild=1) as p:
        p.starmap(do_random_search, rs_params)

    logger.info("All experiments finished :))")

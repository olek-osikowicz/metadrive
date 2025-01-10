from scenario_runner import ScenarioRunner, logger

from itertools import count, product
import random
import time

SAVE_DIR = "/home/olek/Documents/dev/metadrive-multifidelity-data/data/random_search"


def do_random_search(rep, dr, dt, time_budget=60):
    logger.info(f"Starting random search for: {rep = }")
    start_ts = time.time()
    for it in count():
        logger.info(f"Starting iteration {it = }")
        env_seed = random.randint(100, 10**10)
        ScenarioRunner(f"{SAVE_DIR}/{rep}/{it}", env_seed, dr, dt).run_scenario(
            record_gif=True, repeat=True
        )

        if time.time() - start_ts > time_budget:
            logger.info(f"Time elapsed!")
            break

    logger.info(f"Random search finished for: {rep = }!")


if __name__ == "__main__":
    random.seed(2137)

    TIME_BUDGET = 3600  # 60mins
    N_REPETITIONS = 50

    for rep in range(N_REPETITIONS):
        do_random_search(rep, 5, 0.02, TIME_BUDGET)

    logger.info("All experiments finished :))")

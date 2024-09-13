from scenario_runner import ScenarioRunner, logger

import itertools
import random
import time

SAVE_DIR = "olek/data/random_search"


def do_random_search(rep, dr, dt, time_budget=60):
    logger.info(f"Starting random search for: {dr = :3}, {dt = } {rep = }")
    start_ts = time.time()

    while time.time() - start_ts < time_budget:
        env_seed = random.randint(100, 10**10)
        ScenarioRunner(f"{SAVE_DIR}/{rep}", env_seed, dr, dt).run_scenario(
            record_gif=True, repeat=True
        )

    logger.info(f"Random search for: {dr = :3}, {dt = } {rep = } finished!")


if __name__ == "__main__":
    random.seed(2137)

    TIME_BUDGET = 10 * 60  # 10 mins

    dr_range = [5, 10, 15]
    # dr_range = [5, 10, 15, 20]
    dt_range = [0.02, 0.03]
    # dt_range = [0.02, 0.03, 0.04]
    for rep in range(10):
        for dr, dt in itertools.product(dr_range, dt_range):
            do_random_search(rep, dr, dt, TIME_BUDGET)

    logger.info("All experiments finished :))")

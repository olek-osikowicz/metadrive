from scenario_runner import ScenarioRunner, logger

from itertools import count, product
import random
import time

SAVE_DIR = "olek/data/random_search"


def do_random_search(rep, dr, dt, time_budget=60):
    logger.info(f"Starting random search for: {dr = :3}, {dt = } {rep = }")
    start_ts = time.time()
    for it in count():
        logger.info(f"Starting iteration {it =}")
        env_seed = random.randint(100, 10**10)
        ScenarioRunner(f"{SAVE_DIR}/{rep}/{it}", env_seed, dr, dt).run_scenario(
            record_gif=True, repeat=True
        )

        if time.time() - start_ts > time_budget:
            logger.info(f"Time elapsed!")
            break

    logger.info(f"Random search finished for: {dr = :3}, {dt = } {rep = }!")


if __name__ == "__main__":
    random.seed(2137)

    TIME_BUDGET = 10 * 60  # 10 mins

    N_REPETITIONS = 10
    dr_range = [5, 10, 15, 20]
    dt_range = [0.02, 0.03, 0.04]

    for rep, dr, dt in product(range(N_REPETITIONS), dr_range, dt_range):
        do_random_search(rep, dr, dt, TIME_BUDGET)

    logger.info("All experiments finished :))")

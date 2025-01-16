from scenario_runner import ScenarioRunner, logger

from itertools import count, product
import random
import time
import multiprocessing
from multiprocessing import Pool
import pandas as pd
from pathlib import Path
import numpy as np
import sys

sys.path.append("/home/olek/Documents/dev/metadrive-multifidelity-data/notebooks")
from utils.bayesian_optimisation import expected_improvement_iteration
from utils.parse_metadrive import get_scenarios_df

multiprocessing.set_start_method("spawn", force=True)


DATA_DIR = Path("/home/olek/Documents/dev/metadrive-multifidelity-data/data")
BENCHMARKING_DIR = DATA_DIR / "benchmarking"
EI_DIR = DATA_DIR / "expectedimprovement"
SAMPLED_SCENARIOS_DIR = DATA_DIR / "sampled_scenarios"


def get_candidate_solutions():
    return get_scenarios_df(SAMPLED_SCENARIOS_DIR)


def get_training_data(rep=0):

    logger.info(f"Loading repetition {rep} EI data...")
    ei_df = get_scenarios_df(EI_DIR / f"{rep}")
    return ei_df


def do_ei_bayesopt(rep, time_budget=60):
    logger.info(f"Starting random search for: {rep = }")

    candidates = get_candidate_solutions()
    random.seed(rep)
    np.random.seed(rep)

    start_ts = time.time()
    for it in count():
        logger.info(f"Starting iteration {it = }")
        if it < 3:
            dr, dt = 5, 0.02
            seed = random.randint(2**22, 2**31)
            logger.info(f"Random seed: {seed}")
        else:
            train_df = get_training_data(rep)
            dt, dr, seed = expected_improvement_iteration(train_df, candidates)

        logger.info(f"Next scenario to evaluate is {seed=} at fidelity: ({dr=}, {dt=})")

        path = f"{EI_DIR}/{rep}/{it}"
        ScenarioRunner(path, seed, dr, dt, traffic_density=0).run_scenario(
            record_gif=False, repeat=True
        )

        if time.time() - start_ts > time_budget:
            logger.info(f"Time elapsed!")
            break

    logger.info(f"Random search finished for: {rep = }!")


if __name__ == "__main__":

    TIME_BUDGET = 60  # * 10  # 10 mins
    N_REPETITIONS = 10
    N_PROCESSES = 1

    logger.info(
        f"Starting multiprocessing bayesopt HF-EI search! with {N_PROCESSES = }"
    )

    search_params = [(rep, TIME_BUDGET) for rep in range(N_REPETITIONS)]

    with Pool(N_PROCESSES, maxtasksperchild=1) as p:
        p.starmap(do_ei_bayesopt, search_params)

    logger.info("All experiments finished :))")

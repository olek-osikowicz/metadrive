from utils.scenario_runner import ScenarioRunner, logger
from utils.bayesian_optimisation import bayes_opt_iteration, get_random_scenario_seed

from itertools import count, product
import random
import time
import multiprocessing
from multiprocessing import Pool

multiprocessing.set_start_method("spawn", force=True)

import pandas as pd
from pathlib import Path
import numpy as np
import sys

sys.path.append("/home/olek/Documents/dev/metadrive-multifidelity-data/notebooks")
from utils.parse_metadrive import get_scenarios_df

SEARCH_TIME_BUDGET = 60 * 10  # 10 mins
HF_DR = 5
HF_DT = 0.02

SEEDS = {
    "randomsearch": 1,
    "bayesopt_hf_ei": 2,
    "bayesopt_mf_ei": 3,
    "bayesopt_hf_ucb": 4,
    "bayesopt_mf_ucb": 5,
}

DATA_DIR = Path("/home/olek/Documents/dev/metadrive-multifidelity-data/data")

SAMPLED_SCENARIOS_DIR = DATA_DIR / "sampled_scenarios"


def get_candidate_solutions():
    return get_scenarios_df(SAMPLED_SCENARIOS_DIR)


def get_training_data(rep_path):

    # Optionally we could later load the benchmarking data here
    # skipping for now as we start fresh every time
    logger.info(f"Loading training data from: {rep_path}")
    return get_scenarios_df(rep_path)


def do_search(rep, search_type="randomsearch"):

    logger.info(f"Starting {search_type} search for: {rep = }")
    dr, dt = HF_DR, HF_DT

    # calculate random seed from rep and search type
    random_seed = SEEDS[search_type] * 10**6 + rep
    logger.info(f"Setting random seed to: {random_seed}")
    random.seed(random_seed)
    np.random.seed(random_seed)

    #  TODO refactor
    choosing_times = []
    rep_path = DATA_DIR / search_type / str(rep)

    candidates = get_candidate_solutions()

    start_ts = time.perf_counter()
    for it in count():
        logger.info(f"Starting iteration {it = }")

        # choose the next scenario to evaluate given search type
        choose_next_start = time.perf_counter()
        if search_type == "randomsearch" or it < 3:
            env_seed = get_random_scenario_seed(candidates)
        else:
            _, fidelity, aq_type = search_type.split("_")

            if fidelity == "mf":
                raise NotImplementedError("Multifidelity search not implemented yet")

            train_df = get_training_data(rep_path)
            env_seed = bayes_opt_iteration(train_df, candidates, aq_type)

        choosing_time = time.perf_counter() - choose_next_start
        choosing_times.append(choosing_time)
        logger.info(f"Choosing next scenario took: {choosing_time:.2f}s")

        # evaluate next scenarios
        logger.info(
            f"Next scenario to evaluate is {env_seed=} at fidelity: ({dr=}, {dt=})"
        )
        it_path = rep_path / str(it)
        ScenarioRunner(it_path, env_seed, dr, dt, traffic_density=0).run_scenario(
            repeat=True
        )

        if time.perf_counter() - start_ts > SEARCH_TIME_BUDGET:
            logger.info(f"Time elapsed!")
            break

    logger.info(f"{search_type} finished for: {rep = }!")
    with open(rep_path / f"choosing_times.txt", "w") as f:
        f.write("\n".join(map(str, choosing_times)))


if __name__ == "__main__":

    N_REPETITIONS = 50
    N_PROCESSES = 5

    search_types = [
        "bayesopt_hf_ucb",
        "bayesopt_hf_ei",
        "randomsearch",
    ]
    for search_type in search_types:

        logger.info(
            f"Starting {search_type} with {N_PROCESSES = } and {N_REPETITIONS = }"
        )
        search_params = [(rep, search_type) for rep in range(N_REPETITIONS)]
        print(search_params)
        with Pool(N_PROCESSES, maxtasksperchild=1) as p:
            p.starmap(do_search, search_params)

        logger.info(f"Finished {search_type}!")
        time.sleep(5)

    logger.info("All experiments finished :))")

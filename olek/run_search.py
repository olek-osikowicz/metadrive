from utils.scenario_runner import ScenarioRunner, logger
from utils.bayesian_optimisation import get_aqusition_values

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


DATA_DIR = Path("/home/olek/Documents/dev/metadrive-multifidelity-data/data")

SAMPLED_SCENARIOS_DIR = DATA_DIR / "sampled_scenarios"


def get_candidate_solutions():
    return get_scenarios_df(SAMPLED_SCENARIOS_DIR)


def get_training_data(rep_path):

    # Optionally we could later load the benchmarking data here
    # skipping for now as we start fresh every time
    ei_df = get_scenarios_df(rep_path)
    return ei_df


def get_random_scenario_seed(candidates):
    # sample 1 candidate and return the seed
    return candidates.sample(1).index.values[0][-1]


def get_next_scenario_seed_from_aq(aq, candidates):
    if aq.max() == 0:
        logger.info(f"Maximum AQ 0, taking random candidate!")
        env_seed = get_random_scenario_seed(candidates)
    else:
        logger.info(f"Maximum AQ {aq.max():.3f}, taking best candidate!")
        idx_to_evaluate = aq.argmax()
        env_seed = candidates.iloc[[idx_to_evaluate]].index.values[0][-1]
    return env_seed


def do_search(rep, search_type="randomsearch"):

    logger.info(f"Starting {search_type} search for: {rep = }")
    dr, dt = HF_DR, HF_DT

    # calculate random seed from rep and search type
    random_seed = rep + 10**6 * int("".join(str(ord(c)) for c in search_type))
    logger.info(f"Setting random seed to: {random_seed}")
    random.seed(random_seed)
    np.random.seed(random_seed)

    #  TODO refactor
    choosing_times = []
    rep_path = DATA_DIR / search_type / str(rep)

    candidates = get_candidate_solutions()
    start_ts = time.perf_counter()
    for it in count():
        it_path = rep_path / str(it)
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
            aq = get_aqusition_values(train_df, candidates, aq_type)
            env_seed = get_next_scenario_seed_from_aq(aq, candidates)

        choosing_time = time.perf_counter() - choose_next_start
        choosing_times.append(choosing_time)
        logger.info(f"Choosing next scenario took: {choosing_time:.2f}s")

        # evaluate next scenarios
        logger.info(
            f"Next scenario to evaluate is {env_seed=} at fidelity: ({dr=}, {dt=})"
        )
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
    N_PROCESSES = 1

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

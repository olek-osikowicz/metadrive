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

SEARCH_TIME_BUDGET = 60  # * 10  # 10 mins
HF_DR = 5
HF_DT = 0.02


DATA_DIR = Path("/home/olek/Documents/dev/metadrive-multifidelity-data/data")
BENCHMARKING_DIR = DATA_DIR / "benchmarking"
EI_DIR = DATA_DIR / "expectedimprovement"
SAMPLED_SCENARIOS_DIR = DATA_DIR / "sampled_scenarios"


def get_candidate_solutions():
    return get_scenarios_df(SAMPLED_SCENARIOS_DIR)


def get_training_data(rep=0):

    # Optionally we could later load the benchmarking data here
    # skipping for now as we start fresh every time
    logger.info(f"Loading repetition {rep} EI data...")
    ei_df = get_scenarios_df(EI_DIR / f"{rep}")
    return ei_df


def get_random_scenario_seed(candidates):
    # sample 1 candidate and return the seed
    return candidates.sample(1).index.values[0][-1]


def get_next_scenario_seed_from_aq(aq, candidates):
    if aq.max() == 0:
        logger.info(f"Maximum AQ 0, taking random candidate!")
        env_seed = get_random_scenario_seed(candidates)
    else:
        idx_to_evaluate = aq.argmax()
        env_seed = candidates.iloc[[idx_to_evaluate]].index.values[0][-1]
    return env_seed


def do_search(rep, search_type="randomsearch", hf_only=True):

    logger.info(f"Starting {search_type} search for: {rep = } in {hf_only = }")

    if hf_only:
        logger.info("HF only search!")
        dr, dt = HF_DR, HF_DT
    else:
        raise NotImplementedError("Not implemented yet!")

    random.seed(rep)
    np.random.seed(rep)

    candidates = get_candidate_solutions()
    start_ts = time.perf_counter()
    for it in count():
        logger.info(f"Starting iteration {it = }")

        # choose the next scenario to evaluate given search type
        if search_type == "randomsearch" or it < 3:
            env_seed = get_random_scenario_seed(candidates)

        elif search_type == "expectedimprovement":
            train_df = get_training_data(rep)
            aq = get_aqusition_values(train_df, candidates, aq_type="ei")
            env_seed = get_next_scenario_seed_from_aq(aq, candidates)

        # evaluate next scenarios
        logger.info(
            f"Next scenario to evaluate is {env_seed=} at fidelity: ({dr=}, {dt=})"
        )
        it_path = DATA_DIR / search_type / str(rep) / str(it)
        ScenarioRunner(it_path, env_seed, dr, dt, traffic_density=0).run_scenario(
            repeat=True
        )

        if time.perf_counter() - start_ts > SEARCH_TIME_BUDGET:
            logger.info(f"Time elapsed!")
            break

    logger.info(f"Random search finished for: {rep = }!")


if __name__ == "__main__":

    N_REPETITIONS = 2  # 50
    N_PROCESSES = 1

    logger.info(
        f"Starting multiprocessing bayesopt HF-EI search! with {N_PROCESSES = }"
    )
    search_type = "expectedimprovement"  # randomsearch
    search_params = [(rep, search_type) for rep in range(N_REPETITIONS)]

    print(search_params)
    with Pool(N_PROCESSES, maxtasksperchild=1) as p:
        p.starmap(do_search, search_params)

    logger.info("All experiments finished :))")

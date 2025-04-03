from metadrive.engine.logger import get_logger
from utils.scenario_runner import ScenarioRunner
from utils.bayesian_optimisation import (
    SEARCH_FIDELITIES,
    SEARCH_TYPES,
    HDD_PATH,
    bayes_opt_iteration,
    get_training_data,
    random_search_iteration,
    set_seed,
)

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

logger = get_logger()

sys.path.append("/home/olek/Documents/dev/metadrive-multifidelity-data/notebooks")
from utils.parse_metadrive import get_scenarios_df

SMOKETEST = False
SEARCH_BUDGET = 600 if not SMOKETEST else 30
INITIALIZATION_RATIO = 0.90  # run random search for 10% of BayesOpt
N_REPETITIONS = 30 if not SMOKETEST else 2
N_PROCESSES = 5 if not SMOKETEST else 1

SEARCH_DIR = HDD_PATH / "searches"
SEARCH_DIR.mkdir(exist_ok=True)


def do_search(repetition, search_type="randomsearch", fidelity="multifidelity"):

    logger.info(f"Starting {search_type} search for: {repetition = } in {fidelity = }")

    # REPETITION SETUP
    rep_path = SEARCH_DIR / search_type / str(fidelity) / str(repetition)
    set_seed(repetition, search_type, fidelity)

    # set random seed from rep and search type
    current_budget = SEARCH_BUDGET

    for it in count():
        logger.info(f"Starting iteration {it = }")

        match search_type.split("_"):
            case ["randomsearch"]:
                logger.info("Random search iteration!")
                next_seed, next_fid = random_search_iteration(fidelity)

            case ["bayesopt", aq_type]:
                logger.info(f"{aq_type.upper()} Baysian optimisation iteration")
                if current_budget > INITIALIZATION_RATIO * SEARCH_BUDGET:
                    logger.info(f"Still initializing BayesOpt, using RS iteration")
                    next_seed, next_fid = random_search_iteration(fidelity)
                else:
                    logger.info(f"Doing BayesOpt iteration")
                    train_df = get_training_data(rep_path=rep_path)
                    next_seed, next_fid = bayes_opt_iteration(train_df, aq_type, fidelity)
            case _:
                raise ValueError(f"Invalid search type: {search_type}")

        it_path = rep_path / str(it)
        runner = ScenarioRunner(it_path, next_seed, next_fid)
        runner.run_scenario(repeat=True)
        cost = runner.get_evaluation_cost()
        logger.info(f"Running this scenario cost: {cost}")
        current_budget -= cost

        logger.info(f"Current budget: {current_budget}")

        del runner

        if current_budget <= 0:
            logger.info(f"Budget finished!")
            with open(SEARCH_DIR / "checkpoints.txt", "a") as file:
                file.write(f"Search of {rep_path} finished successfully!")

            break


if __name__ == "__main__":

    search_jobs = list(product(range(N_REPETITIONS), SEARCH_TYPES, SEARCH_FIDELITIES))
    logger.info(f"Search jobs: {search_jobs}")

    with Pool(N_PROCESSES, maxtasksperchild=1) as p:
        p.starmap(do_search, search_jobs)

    logger.info("All experiments finished :))")

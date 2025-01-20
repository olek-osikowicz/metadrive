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

SMOKETEST = False

HIGH_FIDELITY = (0.02, 5)

SEARCH_TYPE_SEEDS = {
    "randomsearch": 1,
    "bayesopt_hf_ei": 2,
    "bayesopt_mf_ei": 3,
    "bayesopt_hf_ucb": 4,
    "bayesopt_mf_ucb": 5,
}
BUDGETING_STRATEGY_SEEDS = {
    "wallclock_time": 1,
    "acquire&driving_time": 2,
    "driving_time": 3,
}


DATA_DIR = Path("/home/olek/Documents/dev/metadrive-multifidelity-data/data")
SEARCH_DIR = DATA_DIR / "new_searches_test"


def get_candidate_solutions():
    path = DATA_DIR / "candidate_solutions.csv"
    assert path.exists(), "Candidate solutions not found"

    df = pd.read_csv(path, index_col=0)
    return df


def get_training_data(rep_path: Path) -> pd.DataFrame:
    # Optionally we could later load the benchmarking data here
    # skipping for now as we start fresh every time
    logger.info(f"Loading training data from: {rep_path}")
    return get_scenarios_df(rep_path)


def process_timings(timings: dict) -> dict:
    return {
        "wallclock_time": timings["acquire_time"] + timings["total_time"],
        "acquire&driving_time": timings["acquire_time"] + timings["scenario_time"],
        "driving_time": timings["scenario_time"],
    }


def do_search(rep, search_type="randomsearch", budgeting_strategy="wallclock_time"):

    logger.info(f"Starting {search_type} search for: {rep = } in {budgeting_strategy = }")

    # REPETITION SETUP
    rep_path = SEARCH_DIR / budgeting_strategy / search_type / str(rep)

    # set random seed from rep and search type
    random_seed = rep
    random_seed += 10**4 * SEARCH_TYPE_SEEDS[search_type]
    random_seed += 10**6 * BUDGETING_STRATEGY_SEEDS[budgeting_strategy]
    logger.info(f"Setting random seed to: {random_seed}")
    random.seed(random_seed)
    np.random.seed(random_seed)

    candidates = get_candidate_solutions()
    current_budget = SEARCH_TIME_BUDGET

    for it in count():
        logger.info(f"Starting iteration {it = }")
        timings = {}

        # acquire the next scenario to evaluate given search type
        acquire_ts = time.perf_counter()
        if search_type == "randomsearch" or it < 3:
            env_seed = get_random_scenario_seed(candidates)
            dt, dr = HIGH_FIDELITY
        else:
            _, fidelity, aq_type = search_type.split("_")

            train_df = get_training_data(rep_path)
            use_multifidelity = fidelity == "mf"
            dt, dr, env_seed = bayes_opt_iteration(
                train_df, candidates, aq_type, multifidelity=use_multifidelity
            )

        acquire_time = time.perf_counter() - acquire_ts

        timings["acquire_time"] = acquire_time
        logger.info(f"Acquireing next scenario took: {acquire_time:.2f}s")

        # evaluate next scenario
        logger.info(f"Next scenario to evaluate is {env_seed=} at fidelity: ({dr=}, {dt=})")

        it_path = rep_path / str(it)
        scenario_timings = ScenarioRunner(
            it_path, int(env_seed), dr, dt, traffic_density=0
        ).run_scenario(repeat=True)

        timings.update(scenario_timings)
        timings.update(process_timings(timings))
        # Save timings to file
        with open(it_path / "timings.txt", "w") as f:
            f.write(str(timings))
        # budgeting
        budget_deduction = timings[budgeting_strategy]
        logger.info(f"Current deduction: {budget_deduction:.2f}s in {budgeting_strategy = }")
        current_budget -= budget_deduction

        if current_budget < 0:
            logger.info(f"Time elapsed!")
            break


if __name__ == "__main__":

    N_REPETITIONS = 50 if not SMOKETEST else 1
    N_PROCESSES = 5 if not SMOKETEST else 1
    SEARCH_TIME_BUDGET = 60 * 10 if not SMOKETEST else 30

    search_types = [
        "bayesopt_mf_ucb",
        "bayesopt_mf_ei",
        # "bayesopt_hf_ei",
        # "bayesopt_hf_ucb",
        # "randomsearch",
    ]

    budgeting_strategies = [
        "wallclock_time",
        "acquire&driving_time",
        "driving_time",
    ]
    for budgeting_strategy in budgeting_strategies:

        for search_type in search_types:
            logger.info(
                f"Starting {search_type} with {budgeting_strategy = } {N_PROCESSES = } and {N_REPETITIONS = }"
            )
            search_params = [
                (rep, search_type, budgeting_strategy) for rep in range(N_REPETITIONS)
            ]
            print(search_params)
            with Pool(N_PROCESSES, maxtasksperchild=1) as p:
                p.starmap(do_search, search_params)

            logger.info(f"Finished {search_type} with {budgeting_strategy = } !")
            time.sleep(5)

    logger.info("All experiments finished :))")

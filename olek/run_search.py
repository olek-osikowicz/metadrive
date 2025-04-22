from functools import partial
from metadrive.engine.logger import get_logger
from utils.bayesian_optimisation import SEARCH_FIDELITIES, SEARCH_TYPES, do_search

from itertools import product
import multiprocessing

logger = get_logger()

N_REPETITIONS = 30
N_PROCESSES = 10

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # search_jobs = list(product(range(N_REPETITIONS), SEARCH_TYPES, SEARCH_FIDELITIES))
    logger.info(f"Allow scenario repeat! Experiments")
    search_types = ["bayesopt_ucb"]
    fids = [60, "multifidelity"]
    search_jobs = list(product(range(N_REPETITIONS), search_types, fids))
    logger.info(f"Search jobs: {search_jobs}")

    with multiprocessing.Pool(N_PROCESSES, maxtasksperchild=1) as p:
        p.starmap(do_search, search_jobs)

    logger.info("All experiments finished :))")

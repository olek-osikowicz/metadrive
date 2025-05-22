from functools import partial
from metadrive.engine.logger import get_logger
from utils.bayesian_optimisation import FIDELITY_RANGE, do_search
import numpy as np
from itertools import product
import multiprocessing

logger = get_logger()

N_REPETITIONS = 50
N_PROCESSES = 10


def do_search_wrapper(args):
    return do_search(*args)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # search_jobs = list(product(range(N_REPETITIONS), SEARCH_TYPES, SEARCH_FIDELITIES))
    logger.info(f"Epsilon Fidelity Experiments!")

    fids = FIDELITY_RANGE.copy()
    fids.extend([f"multifidelity_{epsilon:.2f}" for epsilon in np.arange(0, 0.51, 0.05)])

    # Mulifidelity 0.0 should be the same as original MF bayesopt algorithmq
    search_jobs = list(product(range(N_REPETITIONS), ["bayesopt_ucb"], fids))

    search_jobs.extend(product(range(N_REPETITIONS), ["randomsearch"], [60, "multifidelity_0.00"]))

    logger.info(f"Search jobs: {search_jobs} {len(search_jobs) = }")
    with multiprocessing.Pool(N_PROCESSES, maxtasksperchild=1) as p:

        for _ in p.imap_unordered(do_search_wrapper, search_jobs, chunksize=1):
            pass
    logger.info("All experiments finished :))")

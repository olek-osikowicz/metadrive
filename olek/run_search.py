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

    bayesopt_initialization_ratio = 1 - args[0]
    logger.info(
        f"Running search with initialization ratio: {bayesopt_initialization_ratio} {args[1:]}"
    )
    path = f"/media/olek/8TB_HDD/metadrive-data/initialization/{int(args[0] * 100)}-initialization"
    logger.info(f"Search root dir: {path}")
    return do_search(
        *args[1:],
        bayesopt_initialization_ratio=bayesopt_initialization_ratio,
        search_root_dir=path,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)


    logger.info(f"Initialization budget Experiments!")

    initialization_budget = [0.05, 0.15, 0.20]

    search_jobs = list(
        product(
            initialization_budget,
            range(N_REPETITIONS),
            ["bayesopt_ucb"],
            ["multifidelity_0.10"],
        )
    )

    logger.info(f"Search jobs: {search_jobs} {len(search_jobs) = }")
    with multiprocessing.Pool(N_PROCESSES, maxtasksperchild=1) as p:

        for _ in p.imap_unordered(do_search_wrapper, search_jobs, chunksize=1):
            pass

    logger.info("All experiments finished :))")

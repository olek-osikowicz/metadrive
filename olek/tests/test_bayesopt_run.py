from functools import partial
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

import pytest
import numpy as np
from itertools import product
from pathlib import Path
import shutil

from utils.bayesian_optimisation import HDD_PATH, SEARCH_FIDELITIES, SEARCH_TYPES, do_search


@pytest.mark.order(1)
def test_run_search():
    N_REPETITIONS = 2
    N_PROCESSES = 10
    search_jobs = list(product(range(N_REPETITIONS), SEARCH_TYPES, SEARCH_FIDELITIES[-1:]))

    for i in range(2):
        path = Path(f"/tmp/md_test/search_{i}")
        shutil.rmtree(path, ignore_errors=True)
        do_search_partial = partial(do_search, smoketest=True, search_root_dir=path)

        with multiprocessing.Pool(N_PROCESSES, maxtasksperchild=1) as p:
            p.starmap(do_search_partial, search_jobs)


@pytest.mark.order(2)
def test_run_search_results():

    path_a = Path(f"/tmp/md_test/search_0")
    path_b = Path(f"/tmp/md_test/search_1")

    def get_relative_file_paths(base_path):
        return ["/".join(x.parts[-5:]) for x in base_path.rglob("*.json")]

    files_a = get_relative_file_paths(path_a)
    files_b = get_relative_file_paths(path_b)

    def test_uniqueness(files):
        assert len(files) == len(set(files)), "Files are not unique"

    test_uniqueness(files_a)
    test_uniqueness(files_b)

    assert len(files_a) == len(files_b), "Number of files do not match"
    assert sorted(files_a) == sorted(files_b), "Files do not match"

import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

import pytest
import numpy as np
from itertools import product
from pathlib import Path

from utils.bayesian_optimisation import (
    FIDELITY_RANGE,
    SEARCH_FIDELITIES,
    SEARCH_TYPES,
    get_candidate_solutions,
    get_random_scenario_seed,
    get_training_data,
    preprocess_features,
    random_search_iteration,
    regression_pipeline,
    bayes_opt_iteration,
    set_seed,
)


@pytest.fixture(scope="module")  # Call only once per module
def candidate_solutions_df():
    return get_candidate_solutions()


@pytest.fixture(scope="module")
def benchmark_df():
    return get_training_data(benchmark_data=True)


def test_get_candidate_solutions(candidate_solutions_df):
    df = candidate_solutions_df.copy()
    assert not df.empty
    assert len(df) == 100_000


def test_get_benchmark_data(benchmark_df):
    df = benchmark_df.copy()
    assert not df.empty
    assert len(df) == 4_000
    assert "eval.driving_score" in df.columns
    assert df["eval.driving_score"].all()


def assert_features(df):
    assert "def.map_seq.1.id" in df.columns
    assert "def.map_seq.2.id" in df.columns
    assert "def.map_seq.3.id" in df.columns
    assert "def.map_seq.4.id" in df.columns
    assert "def.map_seq.5.id" in df.columns
    assert "def.distance" in df.columns
    assert "def.seed" not in df.columns
    assert df.filter(like="eval").empty


def test_preprocess_benchmark_data(benchmark_df):
    df = benchmark_df.copy()
    df = preprocess_features(df)
    assert "fid.ads_fps" in df.columns, "Fidelity parameter missing"
    assert_features(df)


def test_preprocess_features_with_candidate_solutions(candidate_solutions_df):
    df = candidate_solutions_df.copy()
    df = preprocess_features(df)
    assert_features(df)


def test_regression_pipeline_preprocessing(benchmark_df):
    df = benchmark_df.copy()
    X_train = preprocess_features(df)
    pipes_preprocessing = regression_pipeline(X_train)[:-1]
    values = pipes_preprocessing.fit_transform(X_train)
    # Check if we don't have any NaN -> imputation works
    assert np.isnan(values).any() == False


def test_model_fit(benchmark_df):
    df = benchmark_df.copy()
    X_train = preprocess_features(df)
    pipe = regression_pipeline(X_train)
    y_train = df["eval.driving_score"]
    model = pipe.fit(X_train, y_train)
    assert model.__sklearn_is_fitted__()


def test_mf_bayes_opt_iteration(benchmark_df):
    df = benchmark_df.copy()
    acctual = bayes_opt_iteration(df.copy(), aq_type="ei", fidelity="multifidelity")
    expected = (1049612, 10)
    assert acctual == expected

    acctual = bayes_opt_iteration(df.copy(), aq_type="ucb", fidelity="multifidelity")
    expected = (1007972, 60)
    assert acctual == expected


def test_single_fidelity_bayes_opt_iteration(benchmark_df):

    aq_types = ["ei", "ucb"]
    for fid, aq_type in product(FIDELITY_RANGE, aq_types):

        training_data = benchmark_df.copy().xs(fid, level="fid.ads_fps", drop_level=False)

        next_seed, next_fid = bayes_opt_iteration(training_data, aq_type, fid)
        print(f"Result for {aq_type} in {fid}: {next_seed}, {next_fid}")
        assert fid == next_fid


def test_get_random_candidate(candidate_solutions_df):
    np.random.seed(2137)
    next_seed = get_random_scenario_seed(candidate_solutions_df.copy())
    assert next_seed == 1002891


def test_search_params():
    assert SEARCH_TYPES == ["randomsearch", "bayesopt_ei", "bayesopt_ucb"]
    assert SEARCH_FIDELITIES == [10, 20, 30, 60, "multifidelity"]


def test_set_seed():
    """Check if random state is set properly"""
    set_seed(21, "randomsearch", "multifidelity")
    state = np.random.get_state()
    actual_seed = state[1][0]
    expected = 21 + 10000 + 5000000
    assert actual_seed == expected

    set_seed(0, "bayesopt_ucb", 20)
    state = np.random.get_state()
    actual_seed = state[1][0]
    expected = 0 + 30000 + 2000000
    assert actual_seed == expected


def test_seed_uniqueness():
    """Checks if all the seeds that we generate are unique"""
    seeds_seen = []
    search_params = list(product(range(50), SEARCH_TYPES, SEARCH_FIDELITIES))
    for search in search_params:
        set_seed(*search)
        state = np.random.get_state()
        seed = state[1][0]
        seeds_seen.append(seed)

    assert sorted(seeds_seen) == sorted(list(set(seeds_seen)))


def test_random_search_iteration():

    for i, search_fid in enumerate(SEARCH_FIDELITIES):
        set_seed(i, "randomsearch", search_fid)
        seed_a, fid_a = random_search_iteration(search_fid)

        set_seed(i, "randomsearch", search_fid)
        seed_b, fid_b = random_search_iteration(search_fid)

        assert seed_a == seed_b
        assert fid_a == fid_b

import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

import pytest
from pathlib import Path

from utils.bayesian_optimisation import (
    get_candidate_solutions,
    get_training_data,
    preprocess_features,
    regression_pipeline,
    get_mean_and_std_from_model,
    expected_improvement,
    upper_confidence_bound,
    get_next_scenario_seed_from_aq,
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
    assert 'eval.driving_score' in df.columns 
    assert df['eval.driving_score'].all()

def assert_features(df):
    assert 'def.map_seq.1.id' in df.columns
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
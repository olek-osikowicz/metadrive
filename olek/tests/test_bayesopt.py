import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

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


def test_get_candidate_solutions():
    df = get_candidate_solutions()  
    assert not df.empty
    assert len(df) == 100_000

def test_get_benchmark_data():
    df = get_training_data(benchmark_data=True)  
    assert not df.empty       
    assert 'eval.driving_score' in df.columns 
    assert df['eval.driving_score'].all()
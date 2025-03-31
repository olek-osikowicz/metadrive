import os
import sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from pathlib import Path

from utils.bayesian_optimisation import (
    get_candidate_solutions,
    preprocess_features,
    regression_pipeline,
    get_mean_and_std_from_model,
    expected_improvement,
    upper_confidence_bound,
    get_next_scenario_seed_from_aq,
)

class TestBayesianOptimisation:

    def test_get_candidate_solutions(self):
        df = get_candidate_solutions()  
        assert not df.empty      
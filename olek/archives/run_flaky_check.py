from olek.utils.scenario_runner import ScenarioRunner
from tqdm import tqdm
import itertools

if __name__ == "__main__":

    REPETITIONS_RANGE = range(10)
    SEED_RANGE = range(200)
    # High fidelity only

    for rep, seed in tqdm(itertools.product(REPETITIONS_RANGE, SEED_RANGE)):
        SAVE_DIR = f"data/flaky_check/{rep}"
        ScenarioRunner(SAVE_DIR, seed).run_scenario()

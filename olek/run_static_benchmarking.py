from scenario_runner import ScenarioRunner
from tqdm import tqdm
import itertools

if __name__ == "__main__":

    SEED_RANGE = range(0, 1000)
    DR_RANGE = [5, 10, 15, 20]
    DT_RANGE = [0.02, 0.03, 0.04]

    SAVE_DIR = "data/benchmarking/nocars"
    for seed, dr, dt in tqdm(itertools.product(SEED_RANGE, DR_RANGE, DT_RANGE)):
        ScenarioRunner(SAVE_DIR, seed, dr, dt, traffic_density=0.0).run_scenario()

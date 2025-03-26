from pathlib import Path
from utils.scenario_runner import ScenarioRunner
import itertools
import multiprocessing
import time

multiprocessing.set_start_method("spawn", force=True)

HDD_DIR = Path("/media/olek/2TB_HDD/metadrive-data/sampled_scenarios")


def sample_scenario(seed):

    FPS = 60  # default highfidelity
    ScenarioRunner(HDD_DIR, seed, FPS).run_scenario(dry_run=True)


if __name__ == "__main__":

    start_ts = time.time()
    print("Sampling scenarios...")
    N_SAMPLES = 100_000
    START_SEED = 10**6

    seed_range = [START_SEED + i for i in range(N_SAMPLES)]
    print(seed_range)

    with multiprocessing.Pool(processes=16, maxtasksperchild=1) as pool:
        pool.map(sample_scenario, seed_range)
    print("Scenarios sampled!")

    print(f"Sampling {N_SAMPLES} scenarios took {time.time()-start_ts:.2f}s")

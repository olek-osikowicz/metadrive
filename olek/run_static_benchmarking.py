from pathlib import Path
from utils.scenario_runner import ScenarioRunner
import itertools
import multiprocessing

multiprocessing.set_start_method("spawn", force=True)

HDD_DIR = Path("/media/olek/2TB_HDD/metadrive-data/new_pipeline")


def run_scenario(args):

    seed, rep, fps = args
    save_dir = HDD_DIR / f"{rep}"
    ScenarioRunner(save_dir, seed, fps).run_scenario(record=True, repeat=True, dry_run=False)


if __name__ == "__main__":

    SEED_RANGE = range(1000)
    REPS = 3
    FPS_RANGE = [10, 20, 30, 60]
    multiprocessed = True
    jobs = list(itertools.product(SEED_RANGE, range(REPS), FPS_RANGE))
    print(jobs)

    if multiprocessed:
        with multiprocessing.Pool(processes=16, maxtasksperchild=1) as pool:
            pool.map(run_scenario, jobs)
    else:
        for job in jobs:
            run_scenario(job)

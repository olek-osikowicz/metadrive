from scenario_runner import ScenarioRunner
import itertools

if __name__ == "__main__":

    SAVE_DIR = "olek/data/benchmarking"
    seed_range = range(0, 100)
    dr_range = [5, 10, 15, 20]
    dt_range = [0.02, 0.03, 0.04]
    for seed, dr, dt in itertools.product(seed_range, dr_range, dt_range):
        ScenarioRunner(SAVE_DIR, seed, dr, dt).run_scenario(record_gif=True)

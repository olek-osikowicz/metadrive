from scenario_runner import ScenarioRunner
import itertools

if __name__ == "__main__":

    # ScenarioRunner()
    seed_range = range(0, 1000)
    dr_range = range(5, 21, 5)
    # dt_range = range(0.02, 0.05, 0.001)
    dt_range = [0.02]
    for seed, dr, dt in itertools.product(seed_range, dr_range, dt_range):
        ScenarioRunner(seed, dr, dt).run_scenario()

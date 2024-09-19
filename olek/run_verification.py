from scenario_runner import ScenarioRunner
from tqdm import tqdm
from pathlib import Path

if __name__ == "__main__":

    DATA_DIR = Path("olek/data")
    RS_DIR = DATA_DIR / "random_search"
    VERIFICATION_SAVE_DIR = DATA_DIR / "verification"

    seeds = [path.stem for path in RS_DIR.glob("**/*.json")]
    for seed in tqdm(seeds):
        ScenarioRunner(VERIFICATION_SAVE_DIR, int(seed)).run_scenario()

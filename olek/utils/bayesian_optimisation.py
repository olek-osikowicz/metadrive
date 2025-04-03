import random
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from metadrive.engine.logger import get_logger
from multiprocessing import Pool
from functools import cache, partial
from pathlib import Path
import sys

sys.path.append("/home/olek/Documents/dev/metadrive-multifidelity-data/notebooks")
from utils.parse_metadrive import get_scenarios_df, process_scenario_df  # type: ignore

np.random.seed(0)
logger = get_logger()

HDD_PATH = Path("/media/olek/2TB_HDD/metadrive-data")
assert HDD_PATH.exists()
# current high fidelity is 60 ADS fps.
FIDELITY_RANGE = [10, 20, 30, 60]

SEARCH_TYPES = ["randomsearch", "bayesopt_ei", "bayesopt_ucb"]
SEARCH_FIDELITIES = [*FIDELITY_RANGE, "multifidelity"]


def set_seed(repetition, search_type, fidelity):
    """Set a unique random seed for search experiment parameters"""

    random_seed = repetition
    assert search_type in SEARCH_TYPES
    random_seed += 10**4 * (SEARCH_TYPES.index(search_type) + 1)
    random_seed += 10**6 * (SEARCH_FIDELITIES.index(fidelity) + 1)

    logger.info(f"Setting a random seed: {random_seed}")
    random.seed(random_seed)
    np.random.seed(random_seed)


@cache
def get_candidate_solutions() -> pd.DataFrame:
    candidate_solutions_path = HDD_PATH / "candidate_solutions.parquet"
    assert candidate_solutions_path.exists(), "Candidate solutions don't exist!"
    logger.debug(f"Reading candidate solutions from: {candidate_solutions_path}")
    df = pd.read_parquet(candidate_solutions_path)
    df.index = df.index.rename("def.seed")
    return df


def get_training_data(benchmark_data=True, rep_path: Path | None = None) -> pd.DataFrame:

    # Cached benchmarked data
    if benchmark_data and not rep_path:
        logger.info("Loading benchmarking data")
        dir = HDD_PATH / "basic_traffic" / "0"
        scenario_file = dir / "cache"
        if not scenario_file.exists():
            logger.info("Recreating cache")
            df = get_scenarios_df(dir, multiprocessed=True)
            df.to_json(scenario_file)

        df = pd.read_json(scenario_file)
        df = process_scenario_df(df)

        df = df.set_index(["fid.ads_fps", "def.seed"]).sort_index()
        return df
    else:
        # Load search data
        assert rep_path and rep_path.exists()
        logger.info(f"Loading search data from {rep_path}")
        df = get_scenarios_df(rep_path, multiprocessed=False)
        df = process_scenario_df(df)
        df = df.set_index(["fid.ads_fps", "def.seed"]).sort_index()
        return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    # Drop constant columns, apart from fidelity
    df = df.loc[:, df.nunique(dropna=False) > 1]
    df = df.reset_index()
    # use fidelity and scenario definitions as features
    df = df.loc[:, df.columns.str.startswith("fid.") | df.columns.str.startswith("def.")]
    # remove seed as it's not a feature
    df = df.drop(columns=["def.seed"])
    return df


def regression_pipeline(
    X: pd.DataFrame,
    regressor_cls=RandomForestRegressor,
    cat_features_encoder=OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    ),
    seed=0,
) -> Pipeline:

    categorical_features = X.select_dtypes("object").columns
    numeric_features = X.select_dtypes("number").columns

    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            # Pass through numerical features
            ("Numeric Features", "passthrough", numeric_features),
            # Encode categorical features
            ("Categorical Features", cat_features_encoder, categorical_features),
        ]
    )

    regressor = regressor_cls(random_state=seed)

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
            ("regressor", regressor),
        ],
    )


def expected_improvement(mean, std, best_score):

    # return std
    ei = best_score - mean
    logger.info(f"Maximum EI: {ei.max():.3f}")

    ei = np.maximum(0, ei)
    logger.info(f"Maximum positive EI: {ei.max():.3f}")

    ei = ei + std
    logger.info(f"Maximum EI with uncertainty: {ei.max():.3f}")
    return ei


def upper_confidence_bound(mean, std):
    ucb = 1 - mean
    logger.info(f"Maximum fitness: {ucb.max():.3f}")

    ucb = ucb + std
    logger.info(f"Maximum UCB: {ucb.max():.3f}")
    return ucb


def get_mean_and_std_from_model(model, X_test):
    # Decompose the pipeline
    preprocessing = model[:-1]
    forest = model[-1]

    X_test_processed = preprocessing.transform(X_test)
    tree_predictions = [tree.predict(X_test_processed) for tree in forest.estimators_]
    mean = np.mean(tree_predictions, axis=0)
    std = np.std(tree_predictions, axis=0)

    assert np.allclose(model.predict(X_test), mean), "Predictions are not equal"
    return mean, std


def get_random_scenario_seed(candidates):
    # sample 1 candidate and return the seed
    return candidates.sample(1).index.values[0]


def random_search_iteration(fidelity) -> tuple[int, int]:
    """Performs random search iteration."""
    candidates = get_candidate_solutions()
    next_seed = get_random_scenario_seed(candidates)
    next_fid = fidelity if fidelity != "multifidelity" else random.choice(FIDELITY_RANGE)
    return next_seed, next_fid


def get_next_scenario_seed_from_aq(aq, candidates):
    assert len(aq) == len(candidates), "AQ and candidates must have the same length"
    if aq.max() == 0:
        logger.info(f"Maximum AQ 0, taking random candidate!")
        return get_random_scenario_seed(candidates)
    else:
        idx_to_evaluate = aq.argmax()
        return candidates.index[idx_to_evaluate]


def pick_next_fidelity(
    next_cadidate: pd.DataFrame, scenario_features, trained_model, epsilon=0.01
) -> int:
    """
    Given chosed scenario decide which fidelity is safe to run.
    Returns fidelity.
    """
    logger.info(f"Picking next fidelity!")
    mf_candidates = pd.concat([next_cadidate] * len(FIDELITY_RANGE))
    mf_candidates["fid.ads_fps"] = FIDELITY_RANGE

    mf_X_test = mf_candidates.reset_index()[scenario_features]

    # predict dscore for each fidelity
    predicted_dscore, _ = get_mean_and_std_from_model(trained_model, mf_X_test)

    predictions = dict(zip(FIDELITY_RANGE, predicted_dscore))

    hf_prediction = predictions[max(FIDELITY_RANGE)]
    logger.info(f"Predicted dscore for high fidelity: {hf_prediction:.3f}")
    logger.info(str(predictions))

    # go into increasing fidelity order
    for fid, dscore in predictions.items():
        # maximum absolute error
        error = abs(dscore - hf_prediction)
        logger.info(f"Considering {fid} FPS with predicted {dscore = :.3f}, {error = :.3f}")

        if error < epsilon:
            logger.info(f"Picking fidelity {fid} with dscore error of {error:.3f}")
            return fid

    raise ValueError("No fidelity with acceptable error found")


def bayes_opt_iteration(train_df, aq_type="ei", fidelity="multifidelity") -> tuple[int, int]:
    """
    Performs a single iteration of Bayesian Otpimisation
    Returns next scenario seed, and next fidelity to run.

    """

    logger.info(f"Entering Bayesian Opt Iteration with parameters:")
    logger.info(f"N training samples {len(train_df)}, {aq_type = }, {fidelity = }")
    target_fidelity = fidelity
    if fidelity == "multifidelity":
        target_fidelity = max(FIDELITY_RANGE)

    # PREPARE TRAINING DATA
    X_train = preprocess_features(train_df)
    y_train = train_df["eval.driving_score"]

    if target_fidelity not in train_df.index.get_level_values("fid.ads_fps"):
        logger.warning(f"Target fidelity is not present in training set.")
        logger.warning(f"Will run target fidelity now!")
        return get_random_scenario_seed(get_candidate_solutions()), target_fidelity

    current_best = y_train.xs(target_fidelity).min()
    logger.info(f"Current best score is: {current_best:.3f}")

    # TRAIN THE MODEL
    pipe = regression_pipeline(X_train)
    logger.info(f"Training using {len(X_train.columns)} features")
    # pipe.set_params(regressor__n_jobs=16)
    model = pipe.fit(X_train, y_train)
    logger.debug(f"Model trained")

    # PREPARE TEST DATA
    candidate_scenarios = get_candidate_solutions()
    # Exclude scenarios that have been evaluated (in any fidelity)
    candidate_scenarios = candidate_scenarios[
        ~candidate_scenarios.index.isin(train_df.index.get_level_values("def.seed"))
    ]
    logger.debug(f"Considering next scenario from {len(candidate_scenarios)} candidates.")

    X_test = preprocess_features(candidate_scenarios)
    # test candidates must be casted to target fidelity
    X_test["fid.ads_fps"] = target_fidelity
    X_test = X_test[X_train.columns]

    # PREDICT DSCORE FOR HIGHFIDELITY
    dscore_predictions, std = get_mean_and_std_from_model(model, X_test)
    logger.info(f"Best from model: {dscore_predictions.min():.3f}")

    match aq_type:
        case "ei":
            aq = expected_improvement(dscore_predictions, std, current_best)
        case "ucb":
            aq = upper_confidence_bound(dscore_predictions, std)
        case _:
            raise ValueError("Invalid acquisition function")

    next_seed = int(get_next_scenario_seed_from_aq(aq, candidate_scenarios))
    logger.info(f"Next seed to evaluate: {next_seed}")

    if fidelity != "multifidelity":
        return next_seed, target_fidelity

    logger.debug(f"Multifidelity enabled")

    next_cadidate = candidate_scenarios.loc[[next_seed]]
    next_fidelity = pick_next_fidelity(next_cadidate, X_train.columns, model)
    assert next_fidelity in FIDELITY_RANGE
    return next_seed, next_fidelity

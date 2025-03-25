from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple
import pandas as pd
import numpy as np
from utils.scenario_runner import logger

HIGH_FIDELITY = (0.02, 5)


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    # Drop constant columns, apart from fidelity
    df = df.loc[:, df.nunique() > 1]
    df = df.reset_index()
    df = df.rename(columns={"dt": "fid.dt", "decision_repeat": "fid.decision_repeat"})
    df = df.loc[:, df.columns.str.startswith("fid.") | df.columns.str.startswith("def.")]
    return df


def regression_pipeline(
    X: pd.DataFrame,
    regressor_cls=RandomForestRegressor,
    cat_features_encoder=OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    ),
) -> Pipeline:

    categorical_features = X.select_dtypes("object").columns
    numeric_features = X.select_dtypes("number").columns

    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "Numeric Features",
                SimpleImputer(strategy="constant", fill_value=-1),
                numeric_features,
            ),
            (
                "Categorical Features",
                cat_features_encoder,
                categorical_features,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "regressor",
                regressor_cls(),
            ),
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
    # need to decompose the pipeline
    preprocessing = model[:-1]
    forest = model[-1]

    X_test_processed = preprocessing.transform(X_test)
    tree_predictions = [tree.predict(X_test_processed) for tree in forest.estimators_]
    tree_predictions = np.array(tree_predictions)
    mean = np.mean(tree_predictions, axis=0)
    std = np.std(tree_predictions, axis=0)

    # assert np.equal(mean, model.predict(X_test)).all(), "Predicitons are not equal"
    return mean, std


def get_random_scenario_seed(candidates):
    # sample 1 candidate and return the seed
    return candidates.sample(1).index.values[0]


def get_next_scenario_seed_from_aq(aq, candidates):
    assert len(aq) == len(candidates), "AQ and candidates must have the same length"
    if aq.max() == 0:
        logger.info(f"Maximum AQ 0, taking random candidate!")
        return get_random_scenario_seed(candidates)
    else:
        idx_to_evaluate = aq.argmax()
        return candidates.index[idx_to_evaluate]


def bayes_opt_iteration(
    train_df, candidates, aq_type="ei", multifidelity=False
) -> Tuple[float, int, int]:

    logger.info(
        f"Starting Bayesian optimisation iteration with {aq_type = } and {multifidelity = }"
    )
    # train the model
    X_train = preprocess_features(train_df)
    y_train = train_df["driving_score"]

    current_best = y_train.xs(HIGH_FIDELITY).min()
    logger.info(f"Current best score is: {current_best:.3f}")

    model = regression_pipeline(X_train).fit(X_train, y_train)
    logger.info(f"Model trained!")

    # Filter candidates that we already have data for
    candidates = candidates[~candidates.index.isin(train_df.index.get_level_values("seed"))]
    logger.info(f"Considering next scenario from {len(candidates)} candidates.")

    hf_test = preprocess_features(candidates)
    # project to high fidelity
    hf_test["fid.dt"] = 0.02
    hf_test["fid.decision_repeat"] = 5
    hf_test = hf_test[X_train.columns]

    # find best candidate in high fidelity
    mean, std = get_mean_and_std_from_model(model, hf_test)
    logger.info(f"Best from model: {mean.min():.3f}")
    if aq_type == "ei":
        aq = expected_improvement(mean, std, current_best)
    elif aq_type == "ucb":
        aq = upper_confidence_bound(mean, std)
    else:
        raise ValueError("Invalid acquisition function")

    next_seed = get_next_scenario_seed_from_aq(aq, candidates)

    if not multifidelity:
        return *HIGH_FIDELITY, int(next_seed)

    # Aquire optimal fidelity
    next_cadidate = candidates.loc[[next_seed]]
    fidelity_range = [5, 10, 15, 20]
    mf_candidates = pd.concat([next_cadidate] * len(fidelity_range), ignore_index=True)
    mf_candidates["fid.decision_repeat"] = fidelity_range
    mf_candidates["fid.dt"] = 0.02

    mf_test = mf_candidates[X_train.columns]

    predicted_dscore, _ = get_mean_and_std_from_model(model, mf_test)

    # maximum allowed absolute error
    epsilon = 0.01
    logger.info(
        f"Predicted scores for fidelities: {dict(zip(fidelity_range, predicted_dscore))}"
    )
    # go into reverse order to pick the lowest fidelity, that has acceptable error
    for fid, dscore in list(zip(fidelity_range, predicted_dscore))[::-1]:
        error = abs(dscore - predicted_dscore[0])

        if error < epsilon:
            logger.info(
                f"Picking fidelity {fid} which has predicted dscore error of {error:.3f}"
            )
            return HIGH_FIDELITY[0], fid, int(next_seed)

    raise ValueError("No fidelity found within error threshold")

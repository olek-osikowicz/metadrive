from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import numpy as np
from utils.scenario_runner import logger


def drop_constant_columns(df: pd.DataFrame, verbose=False) -> pd.DataFrame:
    # if a column has one element, it's constant and can be dropped
    for series_name, series in df.items():
        if len(series.map(str).unique()) == 1:
            df = df.drop(series_name, axis=1)
            if verbose:
                logger.info(f"Dropped: {series_name}")

    return df


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index()
    df = drop_constant_columns(df)
    df = df.rename(columns={"dt": "fid.dt", "decision_repeat": "fid.decision_repeat"})
    df = df.loc[
        :, df.columns.str.startswith("fid.") | df.columns.str.startswith("def.")
    ]
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


def get_mean_and_std(X_train, y_train, X_test):
    pipeline = regression_pipeline(X_train)
    pipeline = pipeline.fit(X_train, y_train)

    # need to decompose the pipeline
    preprocessing = pipeline[:-1]
    forest = pipeline[-1]

    X_test_processed = preprocessing.transform(X_test)
    tree_predictions = [tree.predict(X_test_processed) for tree in forest.estimators_]
    tree_predictions = np.array(tree_predictions)
    mean = np.mean(tree_predictions, axis=0)
    std = np.std(tree_predictions, axis=0)

    assert np.equal(mean, pipeline.predict(X_test)).all(), "Predicitons are not equal"
    return mean, std


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


def get_random_scenario_seed(candidates):
    # sample 1 candidate and return the seed
    return candidates.sample(1).index.values[0][-1]


def get_next_scenario_seed_from_aq(aq, candidates):

    assert len(aq) == len(candidates), "AQ and candidates must have the same length"

    if aq.max() == 0:
        logger.info(f"Maximum AQ 0, taking random candidate!")
        env_seed = get_random_scenario_seed(candidates)
    else:
        logger.info(f"Maximum AQ {aq.max():.3f}, taking best candidate!")
        idx_to_evaluate = aq.argmax()
        env_seed = candidates.iloc[[idx_to_evaluate]].index.values[0][-1]

    return env_seed


def bayes_opt_iteration(train_df, candidates, aq_type="ei") -> int:
    """Returns next scenario seed to evaluate from candidates using Bayesian optimization"""

    # Filter out already evaluated candidates
    candidates = candidates[~candidates.index.isin(train_df.index)]

    X_train, X_test = preprocess_features(train_df), preprocess_features(candidates)
    y_train = train_df["driving_score"]

    # use columns that are present in the evaluated scenarios data
    X_test = X_test[X_train.columns]

    current_best = y_train.min()
    logger.info(f"Current best score is: {current_best:.3f}")

    # train the model
    mean, std = get_mean_and_std(X_train, y_train, X_test)
    logger.info(f"Best from model: {mean.min():.3f}")

    if aq_type == "ei":
        aq = expected_improvement(mean, std, current_best)
    elif aq_type == "ucb":
        aq = upper_confidence_bound(mean, std)
    else:
        raise ValueError("Invalid acquisition function")

    return get_next_scenario_seed_from_aq(aq, candidates)


# regression_pipeline(get_training_data())

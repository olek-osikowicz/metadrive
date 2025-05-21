import functools
import multiprocessing
import warnings
import pandas as pd
import math
from pathlib import Path
import orjson
from tqdm import tqdm
from functools import cache
import ast
from tqdm.contrib.concurrent import process_map


def drop_constant_columns(df: pd.DataFrame, verbose=False) -> pd.DataFrame:
    # Identify columns with a single unique value
    constant_columns = []
    for col in df.columns:
        try:
            # Keep NaNs as features
            if df[col].nunique(dropna=False) == 1:
                constant_columns.append(col)
        except TypeError:
            # Skip columns with unhashable types
            warnings.warn(f"Skipping column {col} due to unhashable data.")

    # Drop the constant columns
    for col in constant_columns:
        if verbose:
            print(f"Dropping column {col} with constant value {df[col].unique()}")

    return df.drop(columns=constant_columns)


def get_in_road_percentage(steps_df: pd.DataFrame) -> float:
    ret = steps_df["out_of_road"].value_counts(normalize=True).at[False]
    return ret


def get_n_sidewalk_crashes(steps_df: pd.DataFrame) -> int:
    """Count number of crash episodes to not count same crash multiple times"""
    try:
        # count number of times "crash" becomes True for some time
        n_crashes = steps_df["crash_sidewalk"].diff().value_counts().at[True]

        # need to divide by 2 beacouse diff counts twice
        n_crashes /= 2

        # just in case crash is last episode and we have 3.5 crash episodes make it 4
        n_crashes = math.ceil(n_crashes)
    except KeyError:
        n_crashes = 0

    return n_crashes


def process_steps(steps_infos: list) -> dict:
    """Accepts a list of steps and returns a dict of interesting data"""

    steps_df = pd.DataFrame(steps_infos)
    steps_data = {}
    last_step = steps_df.iloc[-1]
    arrived = last_step["arrive_dest"]
    route_completion = 1.0 if arrived else last_step["route_completion"]

    steps_data = {
        "eval.termination.timeout": last_step["max_step"],
        "eval.termination.arrive_dest": arrived,
        "eval.route_completion": route_completion,
        "eval.in_road_percentage": get_in_road_percentage(steps_df),
        "eval.n_sidewalk_crashes": get_n_sidewalk_crashes(steps_df),
    }

    return steps_data


# ! Problem changing the values here in the analysis can change % error
def calculate_driving_score(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df["eval.driving_score"] = (
            df["eval.route_completion"]
            * df["eval.in_road_percentage"]
            * (0.65) ** df["eval.n_sidewalk_crashes"]
            * (0.60) ** df["eval.n_crashed_vehicles"]
        )
    except KeyError:
        warnings.warn("Error calculating driving score.")
        df["eval.driving_score"] = 0

    return df


# def calculate_driving_score_error(df: pd.DataFrame) -> pd.DataFrame:
#     oracle_ds = df.xs((0.02, 5))["driving_score"]
#     df["driving_score_error"] = (df["driving_score"] - oracle_ds).abs()
#     return df


def add_prefix_to_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_prefix = [
        col for col in df.columns if col.endswith("time") or col.endswith("ts")
    ]
    df = df.rename(columns={col: f"time.{col}" for col in columns_to_prefix})
    return df


def distance_3d(x1, y1, z1, x2, y2, z2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** 0.5


def parse_vehicles_dict(data: dict):
    vehicle_list = list(data.values())
    ret = {"n_vehicles": len(vehicle_list)}

    ego = vehicle_list[0]
    assert ego["type"] == "DefaultVehicle", "Error parsing"

    ret_vehicles = []
    for v in vehicle_list:
        vehicle = {}
        vehicle["distance_to_ego"] = distance_3d(*ego["position"], *v["position"])
        x, y, z = v["position"]
        vehicle["position_x"] = x
        vehicle["position_y"] = y
        vehicle["position_z"] = z
        vehicle["type"] = v["type"]
        vehicle["heading_theta"] = v["heading_theta"]
        vehicle["length"] = v["length"]
        vehicle["width"] = v["width"]
        vehicle["height"] = v["height"]
        vehicle["spawn_road"] = str(v["spawn_road"])
        vehicle["destination"] = str(v["destination"])
        ret_vehicles.append(vehicle)

    ret_vehicles = sorted(ret_vehicles, key=lambda x: x["distance_to_ego"])
    # Label the vehicles, ego is first then enumerate the rest
    ret = {
        (f"vehicle_{i}_{key}" if i > 0 else f"ego_{key}"): value
        for i, vehicle in enumerate(ret_vehicles)
        for key, value in vehicle.items()
    }
    return ret


def normalize_vehicles_data(df: pd.DataFrame) -> pd.DataFrame:
    if "def.vehicles_data" not in df.columns:
        warnings.warn("No BV sequence data found.")
        return df

    vehicles_data = df["def.vehicles_data"].apply(parse_vehicles_dict)
    vehicles_df = pd.json_normalize(vehicles_data).add_prefix("def.vehicles_data.")
    df = pd.concat([df, vehicles_df], axis=1)
    df = df.drop(columns=["def.vehicles_data"])

    return df


# Normalize the 'def.map_seq' column
def normalize_map_data(df: pd.DataFrame) -> pd.DataFrame:
    if "def.map_seq" not in df.columns:
        warnings.warn("No map sequence data found.")
        return df

    map_df = pd.json_normalize(df["def.map_seq"])
    map_df = pd.concat(
        [pd.json_normalize(map_df[col]).add_prefix(f"{col}.") for col in map_df], axis=1
    )

    map_df = map_df.set_index(df.index)
    df = pd.concat([df, map_df.add_prefix("def.map_seq.")], axis=1)
    df = df.drop(columns=["def.map_seq"])
    return df


def test_normalization(df: pd.DataFrame):
    assert "def.vehicles_data" not in df.columns
    assert "def.map_seq" not in df.columns
    for _, seed_df in df.groupby("seed"):

        def_columns = seed_df.columns[seed_df.columns.str.startswith("def")]
        seed_df = seed_df[def_columns]
        seed_df = seed_df.drop(columns="def.max_steps")
        are_unique = seed_df.map(str).nunique(axis=0).eq(1).all()
        assert are_unique, "Scenario definitions are not consistent for a single seed!"


def process_scenario_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        warnings.warn("Empty dataframe.")
        return df
    df = normalize_map_data(df)
    df = normalize_vehicles_data(df)
    df = df.drop(columns=["def.env_config", "def.reset_info"])

    df = calculate_driving_score(df)
    df = add_prefix_to_time_columns(df)
    return df


@cache
def _read_scenario_data(file_path: Path, eval_type):
    with open(file_path, "rb") as f:  # open in binary mode
        scenario_data = orjson.loads(f.read())

    if scenario_data["eval.steps_infos"]:
        steps_infos = scenario_data.pop("eval.steps_infos")
        scenario_data.update(process_steps(steps_infos))
    return scenario_data


def get_scenarios_df(dir: Path, eval_type="benchmark", multiprocessed=False) -> pd.DataFrame:

    if not dir.exists():
        warnings.warn("Data directory doesn't exist.")
        return pd.DataFrame()

    def get_iteration_from_path(x: Path) -> int:
        return int(x.parts[-2])

    paths = sorted(dir.rglob("*.json"), key=get_iteration_from_path)

    if multiprocessed:
        df = pd.DataFrame(
            process_map(
                functools.partial(_read_scenario_data, eval_type=eval_type),
                paths,
                max_workers=8,
                chunksize=math.ceil(len(paths) / 100),
            )
        )

    else:
        df = pd.DataFrame(
            [_read_scenario_data(file_path, eval_type) for file_path in tqdm(paths)]
        )
    # print("Processing dataframe...")
    # df = process_scenario_df(df, eval_type)
    return df


def get_map_sequence_columns() -> list[str]:

    N_BLOCKS = 5

    columns_order = ["def.spawn_lane_index"]
    for n in range(1, N_BLOCKS + 1):
        if n > 1:
            columns_order.append(f"def.map_seq.{n}.pre_block_socket_index")
        columns_order.append(f"def.map_seq.{n}.id")
    columns_order.extend(
        ["def.vehicles_data.destination.0", "def.vehicles_data.destination.1"]
    )

    return columns_order


if __name__ == "__main__":
    DATA_DIR = Path("data/nocars")
    assert DATA_DIR.exists(), "Data dir not found"
    scenarios_df = get_scenarios_df(DATA_DIR)
    print(scenarios_df.head())

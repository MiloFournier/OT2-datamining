import pandas as pd
import numpy as np
from typing import Optional, List, Callable, Any, Union, Dict
from itertools import product
from statistics import mean
from pathlib import Path
import gzip
import os
import re


def read_ds_gzip(path: Optional[Path] = None, ds: str = "TRAIN") -> pd.DataFrame:
    """Args:
        path (Optional[Path], optional): the path to read the dataset file. Defaults to /kaggle/input/the-insa-starcraft-2-player-prediction-challenge/{ds}.CSV.gz.
        ds (str, optional): the part to read (TRAIN or TEST), to use when path is None. Defaults to "TRAIN".

    Returns:
        pd.DataFrame:
    """
    with gzip.open(
        f"/kaggle/input/the-insa-starcraft-2-player-prediction-challenge/{ds}.CSV.gz"
        if path is None
        else path
    ) as f:
        max_actions = max((len(str(c).split(",")) for c in f.readlines()))
        f.seek(0)
        _names = ["battleneturl", "played_race"] if "TRAIN" in ds else ["played_race"]
        _names.extend(range(max_actions - len(_names)))
        return pd.read_csv(f, names=_names, dtype=str)


def first_nan_occurrence(row):
    return row.first_valid_index() if row.isna().all() else row.isna().idxmax()


def max_t_value(row):
    t_values = []
    # Iterate over each column (excluding the first two columns, e.g., 'battleneturl' and 'played_race')
    for value in row[2:]:  # Skip the first two columns
        # Find values starting with 't' followed by digits
        match = re.match(r"t(\d+)", str(value))
        if match:
            t_values.append(
                int(match.group(1))
            )  # Convert to integer and append to the list

    # Return the maximum value found or NaN if no 't' values exist
    return max(t_values, default=None)


def count_hotkeys(row, pattern):
    hotkeys = [key for key in row if isinstance(key, str) and re.match(pattern, key)]
    game_length = row["max_t_value"]
    return len(hotkeys) / game_length


def count_actions_before_t(row, value):
    # Convert the row to a list
    row_list = row.tolist()
    try:
        # Find the index of the first occurrence of the specified value (e.g., "t5")
        t_index = row_list.index(f"{value}")
        return t_index
    except ValueError:
        # If the value is not found, return the total number of elements
        return len(row_list)


def max_actions_between_t(row):
    # Convert the row to a list
    row_list = row.tolist()
    # Find all indices of 't' markers (e.g., t5, t10, t15)
    t_indices = [
        i
        for i, val in enumerate(row_list)
        if isinstance(val, str) and val.startswith("t") and val[1:].isdigit()
    ]
    # Initialize the maximum difference
    max_actions = 0
    # Calculate the number of actions between consecutive 't' markers
    for i in range(1, len(t_indices)):
        actions_between = t_indices[i] - t_indices[i - 1] - 1
        max_actions = max(max_actions, actions_between)
    return max_actions


def min_actions_between_t(row):
    # Convert the row to a list
    row_list = row.tolist()
    # Find all indices of 't' markers (e.g., t5, t10, t15)
    t_indices = [
        i
        for i, val in enumerate(row_list)
        if isinstance(val, str) and val.startswith("t") and val[1:].isdigit()
    ]
    # Initialize the minimum difference as a large number
    min_actions = float("inf")
    # Calculate the number of actions between consecutive 't' markers
    for i in range(1, len(t_indices)):
        actions_between = t_indices[i] - t_indices[i - 1] - 1
        min_actions = min(min_actions, actions_between)

    # If no 't' markers are found or only one 't' marker exists, return 0
    if min_actions == float("inf"):
        return 0
    return min_actions


def calculate_player_avg_stats(df, battleneturl):
    player_df = df[df["battleneturl"] == battleneturl]

    if player_df.empty:
        return f"No player found with battleneturl: {battleneturl}"

    stats_columns = [
        "max_t_value",
        "action_per_sec",
        "played_protoss",
        "played_terran",
        "played_zerg",
        "actions_before_t5",
        "actions_before_t10",
        "actions_before_t30",
    ]

    avg_stats = player_df[stats_columns].mean()

    return avg_stats

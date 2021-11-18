from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Bodyparts = str
Coordinate = str
Model = str
DLCKey = Tuple[Model, Bodyparts, Coordinate]


def remove_scorer(data: pd.DataFrame):
    newnames = list(map(lambda c: (c[1], c[2]), data.keys()))
    idx = pd.MultiIndex.from_tuples(newnames)
    return data.set_axis(idx, axis=1)


def contains(pattern: str) -> Callable:
    def inner(key: DLCKey) -> bool:
        return pattern in key

    return inner


def replace_low_likelihood_nan(data: pd.DataFrame,
                               threshold: float) -> NDArray[np.float_]:
    likelihood_over_threshold = data > threshold
    return np.array(likelihood_over_threshold[likelihood_over_threshold])


def as_dataframe(array: NDArray[np.float_],
                 idx_list: List[Tuple[Bodyparts, Coordinate]]) -> pd.DataFrame:
    d = pd.DataFrame(data=array)
    idx = pd.MultiIndex.from_tuples(idx_list)
    d.set_axis(idx, axis=1, inplace=True)
    return d


def create_dataframe_with_nan(data: pd.DataFrame,
                              nan_matrix: NDArray[np.float_],
                              axis: str) -> pd.DataFrame:
    axis_idx = list(filter(contains(axis), data.keys()))
    return as_dataframe(np.array(data[axis_idx]) * nan_matrix, axis_idx)


def as_output_filename(h5_path: Path, data_name: str):
    parent = h5_path.parents[0]
    stem = h5_path.stem.split("DLC")[0]
    filepath_without_extension = parent.joinpath("interpolated_data").joinpath(
        stem)
    return str(filepath_without_extension) + data_name + ".h5"


if __name__ == '__main__':
    # TODO: Add support for specifying the data to be read
    # and the directory to be output by command line arguments
    AXIS = ["x", "y"]

    h5 = "path/to/analyzed/data"
    tracked_data = remove_scorer(pd.DataFrame(pd.read_hdf(h5)))

    likelihood_idx = list(filter(contains("likelihood"), tracked_data.keys()))
    nan_matrix = replace_low_likelihood_nan(tracked_data[likelihood_idx], 0.9)

    contains_nan = list(
        map(lambda ax: create_dataframe_with_nan(tracked_data, nan_matrix, ax),
            AXIS))
    data_contains_nan = pd.concat(contains_nan,
                                  axis=1).sort_index(axis=1, inplace=False)
    if data_contains_nan is None:
        raise ValueError("Failed to create dataframe.")

    data_interpolated = data_contains_nan.astype("float64").interpolate()

    if data_interpolated is None:
        raise ValueError("Failed to create dataframe.")

    output_path_nan = as_output_filename(Path(h5), "_outlitter_to_nan")
    output_path_interpolated = as_output_filename(Path(h5), "_interpolated")

    data_contains_nan.to_hdf(output_path_nan, "key")
    data_interpolated.to_hdf(output_path_interpolated, "key")

"""Fit ellipse to the points and calculate its area."""
from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from skimage.measure import EllipseModel

Point = np.float_
Coordinate = float
MajorAxis = float
MinorAxis = float
Theta = float
EllipseParams = Tuple[Coordinate, Coordinate, MajorAxis, MinorAxis, Theta]
Area = float
Color = Tuple[int, int, int]
DLCKey = Tuple[str, str, str]


def extract_bodypart_coordinate(data: DataFrame, bodypart: str) -> DataFrame:
    """Return a dataframe that contains coordinates of a given bodypart.

    Parameters
    ----------
    data: DataFrame
        A Dataframe that contains coordinates of bodyparts.
    bodypart: str
        Name of a bodypart.

    Returns
    -------
    extracted: DataFrame
        A Dataframe that contains coordinates of the given bodyparts.
    """
    bodypart_coordinates = list(
        filter(lambda key: bodypart in key[0], data.keys()))
    return data[bodypart_coordinates]


def reshape2fittable(data: DataFrame) -> NDArray[np.float_]:
    """Reshape a dataframe to the shape that can be passed to `fit_ellipse`.

    Parameters
    ----------
    data: DataFrame
        A dataframe that contains tracked points for each bodypart.

    Returns
    -------
    fittable: NDArray[3, np.float_]
        3D array that can be passed to `fit_ellipse`
    """
    nrow, ncol = data.shape
    return np.array(data).reshape(nrow, -1, 2).astype(np.float_)


def fit_ellipse(points: NDArray[Point],
                minpoints: int) -> Optional[EllipseParams]:
    """Fit ellipse to a given points.

    This function fit an ellipse to the given points.
    Return None, if the number of given points less than `minpoints`.

    Parameters
    ----------
    points: NDArray[2, Point]
        A 2D array of point coordinates.
    minpoint: int
        Minimum number of points used to fit ellipse.

    Returns
    -------
    params: Optional[EllipseParams]
        Estimated parameters of the ellipse or None.

    Notes
    -----
    1. The return contains coordinates of center coordinate of x and y,
       length of major and minor axis, and radian of the ellipse.
    """
    points_ = points[~np.isnan(points).any(axis=1), :]
    if len(points) <= 3:
        return None
    m = EllipseModel()
    m.estimate(points_)
    return tuple(m.params)


def fit_ellipses(data: NDArray[np.float_],
                 minpoint: int) -> List[EllipseParams]:
    """Fit ellipse to the given list of points

    Parameters
    ----------
    points: NDArray[3, Point]
        A 3D array of point coordinates.
    minpoint: int
        Minimum number of points used to fit ellipse.

    Returns
    -------
    params: List[Optional[EllipseParams]]
        List of estimated parameters of the ellipse or None.

    """
    T, _, _ = data.shape
    return list(map(lambda t: fit_ellipse(data[t], minpoint), range(T)))


def calculate_ellipse_area(params: EllipseParams) -> Area:
    """Calculate ellipse area.

    Parametes
    ---------
    params: EllipseParams
        Parameters of an ellipse.

    Returns
    -------
    area: Area
        Area of the ellipse.
    """
    _, _, a, b, _ = params
    return np.pi * a * b


def draw_ellipse(frame: NDArray, params: EllipseParams, color: Color,
                 **kwargs) -> NDArray:
    """Draw an ellipse on a given frame.

    Parameters
    ----------
    frame: NDArray
        An image to draw an ellipse.
    params: EllipseParams
        Parameters of an ellipse.
    color: Color
        Color of the ellipse to draw.

    Returns
    -------
    frame_: NDArray
        An image drawn with the ellipse.
    """
    # To avoid dide effects, make a copy and draw on it.
    frame_ = frame.copy()
    xcenter, ycenter, major, minor, theta = params
    angle = 180. * theta / np.pi
    cv2.ellipse(frame_, ((xcenter, ycenter), (2 * major, 2 * minor), angle),
                color, **kwargs)
    return frame_


if __name__ == '__main__':
    # TODO: Add support for specifying the data to be read
    # and the directory to be output by command line arguments
    import pandas as pd

    BODYPARTS = ["pupil", "eye"]
    MINPOINTS = 4

    h5 = "path/to/interpolated-data"

    tracked_data = pd.DataFrame(pd.read_hdf(h5))
    areasizes: List[List[np.float_]] = []

    for bodypart in BODYPARTS:
        bodypart_data = reshape2fittable(
            extract_bodypart_coordinate(tracked_data, bodypart))
        ellipse_params = fit_ellipses(bodypart_data, MINPOINTS)
        areasize = list(map(calculate_ellipse_area, ellipse_params))
        areasizes.append(areasize)

    areasizes = np.array(areasizes)
    area_data = pd.DataFrame(areasizes.T, columns=BODYPARTS)

    print(area_data)
    # TODO: implement data output

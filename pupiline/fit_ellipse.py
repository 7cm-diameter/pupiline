from os import listdir
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.measure import EllipseModel


def fit_ellipse(
    points: npt.NDArray[np.float_],
) -> tuple[float, float, float, float, float]:
    points = points[~np.isnan(points).any(axis=1), :]
    m = EllipseModel()
    m.estimate(points)
    x, y, a, b, t = m.params
    return x, y, a, b, t


def calc_ellipse_area(params: tuple[float, float, float, float, float]) -> float:
    _, _, a, b, _ = params
    return np.pi * a * b


def draw_ellipse(
    frame: npt.NDArray,
    params: tuple[float, float, float, float, float],
    color: tuple[int, int, int],
    thickness: int,
):
    xc, yc, a, b, theta = params
    angle = 180.0 * theta / np.pi
    cv2.ellipse(
        frame, ((xc, yc), (2.0 * a, 2.0 * b), angle), color, thickness=thickness
    )


def is_coordinate(key: tuple[str, str, str]):
    return "x" in key or "y" in key


def extract_key_of_bodyparts(
    data: pd.DataFrame, bodyparts: str
) -> list[tuple[str, str, str]]:
    coords = list(filter(is_coordinate, data.keys()))
    return list(filter(lambda key: bodyparts in key[0], coords))


def reshape2fittable(data: pd.DataFrame) -> npt.NDArray[np.float_]:
    nrow, ncol = data.shape
    return np.array(data).reshape(nrow, -1, 2)


def as_output_filename(video_path: Path):
    parent = video_path.parents[1]
    filepath_without_extension = parent.joinpath("area").joinpath(video_path.stem)
    return str(filepath_without_extension) + ".csv"


def is_marked(
    frame: npt.NDArray,
    position: tuple[int, int],
    color_range: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> int:
    x, y = position
    color = frame[x, y, :]
    lcolor, ucolor = color_range
    for comp in zip(color, lcolor, ucolor):
        c, l, u = comp
        if not (l <= c and c <= u):
            return 0
    return 1


def choose_video(pattern: str, paths: list[Path]) -> Optional[str]:
    matched = list(map(str, filter(lambda p: pattern in str(p), paths)))
    for m in matched:
        if not "ellipse" in m:
            return m
    return None


if __name__ == "__main__":
    from shutil import move
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--create-video", "-c", action="store_true")
    parser.add_argument("--show-video", "-s", action="store_true")
    parser.add_argument("--target-bodypart", "-t", type=str, default="pupil")
    args = parser.parse_args()

    create_video = args.create_video
    show_video = args.show_video
    target_bodypart = args.target_bodypart

    csvs = list(
        map(
            lambda filename: Path("./data/tracked/").joinpath(filename),
            listdir("./data/tracked"),
        )
    )

    videos = list(
        map(
            lambda filename: Path("./data/video").joinpath(filename),
            listdir("./data/video"),
        )
    )

    for csv in csvs:
        pattern = csv.stem.split("DLC")[0]
        video = choose_video(pattern, videos)
        print(f"start processing {video}")
        tracked_data = pd.read_csv(csv, header=[1, 2])
        cap = cv2.VideoCapture(str(video))
        nframe = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if create_video and video is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            writer = cv2.VideoWriter(
                f"./data/video/{Path(video).stem}-ellipse.MP4",
                fourcc,
                fps,
                (width, height),
            )

        pupil_keys = extract_key_of_bodyparts(tracked_data, target_bodypart)
        pupil_data = reshape2fittable(tracked_data[pupil_keys])
        results: list[tuple[float, float, float, int]] = []

        try:
            for i in range(nframe - 1):
                if i % 5000 == 0:
                    print(f"Processing {i}-th frame")
                ret, frame = cap.read()
                pupil_params = fit_ellipse(pupil_data[i])
                pupil_area = calc_ellipse_area(pupil_params)
                pupil_x, pupil_y, _, _, _ = pupil_params
                cs_on = is_marked(frame, (10, 10), ((0, 0, 235), (20, 20, 255)))
                results.append((pupil_area, pupil_x, pupil_y, cs_on))
                if show_video:
                    draw_ellipse(frame, pupil_params, (0, 255, 255), 1)
                    # draw_ellipse(frame, eyelid_params, (0, 255, 0), 1)
                if create_video and video is not None:
                    writer.write(frame)
                if show_video:
                    cv2.imshow("video", frame)
                    if cv2.waitKey(1) % 0xFF == ord("q"):
                        cv2.destroyAllWindows()
                        cap.release()
                        raise Exception()
            if create_video and video is not None:
                writer.release()
                cv2.destroyAllWindows()
            output = pd.DataFrame(
                results,
                columns=[
                    f"{target_bodypart}-area",
                    f"{target_bodypart}-x",
                    f"{target_bodypart}-y",
                    "cs",
                ],
            )
            output_path = as_output_filename(Path(csv))
            output.to_csv(output_path, index=False)
            move(csv, Path("data/analyzed").joinpath(csv.name))
        except:
            print(f"Failed to analyze {csv}.")

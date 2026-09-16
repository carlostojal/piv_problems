from pathlib import Path

import cv2
import numpy as np

BLOCK_WIDTH = 15.8
BLOCK_HEIGHT = 9.6

def build_left_object_points(
    rows=4, cols=6, block_width=BLOCK_WIDTH, block_height=BLOCK_HEIGHT
):
    points = []
    y = 0
    z = rows * block_height
    for _ in range(rows-1):
        x = cols * block_width
        for _ in range(cols-1):
            points.append([x, y, z])
            x -= block_width
        z -= block_height
    return np.array(points, dtype=np.float32)


def build_right_object_points(
    rows=4,
    cols=8,
    block_width=BLOCK_WIDTH,
    block_height=BLOCK_HEIGHT,
):
    points = []
    x = 0
    z = rows * block_height
    for _ in range(rows-1):
        y = 2 * block_width
        for _ in range(cols-1):
            points.append([x, y, z])
            y += block_width
        z -= block_height
    return np.array(points, dtype=np.float32)


def local_calibration(object_points, image_points, image_size):
    width, height = image_size
    camera_matrix = np.array(
        [[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    return cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        camera_matrix,
        None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )


def save_calibration(
    directory,
    camera_matrix,
    distortion,
    rotation_vectors,
    translation_vectors,
    frame_indices,
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "camera_matrix.npy", camera_matrix)
    np.save(directory / "distortion_coefficients.npy", distortion)
    np.save(directory / "rotation_vectors.npy", np.array(rotation_vectors))
    np.save(directory / "translation_vectors.npy", np.array(translation_vectors))
    np.save(directory / "frame_indices.npy", np.array(frame_indices, dtype=np.int64))


def load_calibration(directory):
    directory = Path(directory)
    return (
        np.load(directory / "camera_matrix.npy"),
        np.load(directory / "distortion_coefficients.npy"),
        np.load(directory / "rotation_vectors.npy"),
        np.load(directory / "translation_vectors.npy"),
        np.load(directory / "frame_indices.npy"),
    )

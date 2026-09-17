import cv2
import numpy as np

from calibration_utils import (
	build_right_object_points,
	build_left_object_points,
	local_calibration,
	save_calibration,
)

FRAME_STEP = 1
GRID_ROWS = 4
GRID_COLS = 6

# capture video
cap = cv2.VideoCapture("data/lego_twogrids.mp4")

done = False
i = 0

# build world points and image points
objp = []
objpoints = []
imgpoints = []
frame_indices = []

objp = build_left_object_points(GRID_ROWS, GRID_COLS)

print(f"{len(objp)} points")

while not done:
	ret, frame = cap.read()
	if not ret:
		done = True
		break

	if i % FRAME_STEP == 0:

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect the chessboard corners
		ret, corners = cv2.findChessboardCorners(
			gray, (GRID_COLS-1, GRID_ROWS-1)
		)

		# build object and image points
		if ret:
			objpoints.append(np.array(objp, dtype=np.float32))
			imgpoints.append(corners)
			frame_indices.append(i)
	i += 1

cap.release()

# run calibration
height, width = gray.shape
ret, mtx, dist, rvecs, tvecs = local_calibration(
	objpoints, imgpoints, (width, height)
)

# print calibration parameters
print(f"K={mtx}")

# save calibration parameters in files
save_calibration("data/calib_2d", mtx, dist, rvecs, tvecs, frame_indices)

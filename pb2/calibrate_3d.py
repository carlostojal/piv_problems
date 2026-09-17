import cv2
import numpy as np

from calibration_utils import (
	build_left_object_points,
	build_right_object_points,
	local_calibration,
	save_calibration,
)

FRAME_STEP = 1
LEFT_GRID_ROWS = 4
LEFT_GRID_COLS = 6
RIGHT_GRID_ROWS = 4
RIGHT_GRID_COLS = 8

# capture video
cap = cv2.VideoCapture("data/lego_twogrids.mp4")

done = False
i = 0

# build world points and image points
left_objp = []
right_objp = []
objpoints = []
imgpoints = []
frame_indices = []

left_objp = build_left_object_points(LEFT_GRID_ROWS, LEFT_GRID_COLS)
right_objp = build_right_object_points(RIGHT_GRID_ROWS, RIGHT_GRID_COLS)

print(f"left points: {left_objp.shape}")
print(f"right points: {right_objp.shape}")

while not done:
	ret, frame = cap.read()
	if not ret:
		done = True
		break

	if i % FRAME_STEP == 0:

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		left_found, left_corners = cv2.findChessboardCorners(
			gray, (LEFT_GRID_COLS - 1, LEFT_GRID_ROWS - 1)
		)
		right_found, right_corners = cv2.findChessboardCorners(
			gray, (RIGHT_GRID_COLS - 1, RIGHT_GRID_ROWS - 1)
		)

		if left_found and right_found:
			objpoints.append(np.concatenate([left_objp, right_objp]))
			imgpoints.append(np.concatenate([left_corners, right_corners]))
		elif left_found:
			objpoints.append(left_objp)
			imgpoints.append(left_corners)
		elif right_found:
			objpoints.append(right_objp)
			imgpoints.append(right_corners)


		if left_found or right_found:
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
save_calibration("data/calib_3d", mtx, dist, rvecs, tvecs, frame_indices)

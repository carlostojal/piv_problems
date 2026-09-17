
import cv2
import numpy as np

from calibration_utils import (
	build_left_object_points,
	build_right_object_points,
	load_calibration,
)


object_points = {
	"left": build_left_object_points(),
	"right": build_right_object_points(),
}
calibrations = {
	"2D": load_calibration("data/calib_2d"),
	"3D": load_calibration("data/calib_3d"),
}
pose_indices = {
	name: {frame: index for index, frame in enumerate(frame_indices)}
	for name, (_, _, _, _, frame_indices) in calibrations.items()
}
grid_sizes = {"left": (5, 3), "right": (7, 3)}
error_sums = {
	name: {board: 0.0 for board in object_points} for name in calibrations
}
error_counts = {
	name: {board: 0 for board in object_points} for name in calibrations
}

cap = cv2.VideoCapture("data/lego_twogrids.mp4")
frame_index = 0
while True:
	ret, frame = cap.read()
	if not ret:
		break

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	detections = {}
	for board, grid_size in grid_sizes.items():
		found, corners = cv2.findChessboardCorners(gray, grid_size)
		if found:
			detections[board] = corners.reshape(-1, 2)
			cv2.drawChessboardCorners(frame, grid_size, corners, found)

	for name, (camera_matrix, distortion, rvecs, tvecs, _) in calibrations.items():
		pose_index = pose_indices[name].get(frame_index)
		if pose_index is None:
			continue
		color = (255, 0, 0) if name == "2D" else (0, 0, 255)
		for board, points in object_points.items():
			projected, _ = cv2.projectPoints(
				points,
				rvecs[pose_index],
				tvecs[pose_index],
				camera_matrix,
				distortion,
			)
			if board in detections:
				difference = detections[board] - projected.reshape(-1, 2)
				error_sums[name][board] += float(np.sum(difference ** 2))
				error_counts[name][board] += len(points)
			for x, y in projected.reshape(-1, 2):
				cv2.circle(frame, (int(x), int(y)), 4, color, -1)

	cv2.imshow("Reprojection", frame)
	if cv2.waitKey(30) & 0xFF == ord("q"):
		break
	frame_index += 1

cap.release()
cv2.destroyAllWindows()

for name in calibrations:
	all_squared_errors = 0.0
	all_points = 0
	for board in object_points:
		count = error_counts[name][board]
		if not count:
			continue
		rms = np.sqrt(error_sums[name][board] / count)
		print(f"{name} {board} RMS pixel error: {rms:.4f}")
		all_squared_errors += error_sums[name][board]
		all_points += count
	if all_points:
		print(
			f"{name} RMS pixel error: "
			f"{np.sqrt(all_squared_errors / all_points):.4f}"
		)
	else:
		print(f"{name} RMS pixel error: no valid detections")

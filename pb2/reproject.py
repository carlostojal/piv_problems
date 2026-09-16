
import cv2

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

cap = cv2.VideoCapture("data/lego_twogrids.mp4")
frame_index = 0
while True:
	ret, frame = cap.read()
	if not ret:
		break

	for name, (camera_matrix, distortion, rvecs, tvecs, _) in calibrations.items():
		pose_index = pose_indices[name].get(frame_index)
		if pose_index is None:
			continue
		colors = {
			"left": (255, 120, 0) if name == "2D" else (0, 120, 255),
			"right": (255, 0, 0) if name == "2D" else (0, 0, 255),
		}
		for board, points in object_points.items():
			projected, _ = cv2.projectPoints(
				points,
				rvecs[pose_index],
				tvecs[pose_index],
				camera_matrix,
				distortion,
			)
			for x, y in projected.reshape(-1, 2):
				cv2.circle(frame, (int(x), int(y)), 4, colors[board], -1)

	cv2.imshow("Reprojection", frame)
	if cv2.waitKey(30) & 0xFF == ord("q"):
		break
	frame_index += 1

cap.release()
cv2.destroyAllWindows()

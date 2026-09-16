import cv2
import numpy as np
from pathlib import Path

FRAME_STEP = 1
BLOCK_WIDTH = 15.8
BLOCK_HEIGHT = 9.6

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

# build left grid
x = 0
y = (GRID_COLS) * BLOCK_WIDTH
z = (GRID_ROWS) * BLOCK_HEIGHT
for z_idx in range(GRID_ROWS-1):
	for y_idx in range(GRID_COLS-1):
		objp.append([x, y, z])
		y -= BLOCK_WIDTH
	z -= BLOCK_HEIGHT

print(f"{len(objp)} points")

while not done:
	ret, frame = cap.read()
	if not ret:
		done = True
		break

	if i % FRAME_STEP == 0:

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (GRID_COLS-1,GRID_ROWS-1))

		# build object and image points
		if ret:
			objpoints.append(np.array(objp, dtype=np.float32))
			imgpoints.append(corners)
	i += 1

cap.release()

# run calibration
height, width = gray.shape
camera_matrix = np.array(
	[[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]],
	dtype=np.float64,
)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
	objpoints,
	imgpoints,
	(width, height),
	camera_matrix,
	None,
	flags=cv2.CALIB_USE_INTRINSIC_GUESS,
)

# print calibration parameters
print(f"K={mtx}")

# save calibration parameters in files
calibration_dir = Path("data/calib_2d")
calibration_dir.mkdir(parents=True, exist_ok=True)
np.save(calibration_dir / "camera_matrix.npy", mtx)
np.save(calibration_dir / "rotation_vectors.npy", np.array(rvecs))
np.save(calibration_dir / "translation_vectors.npy", np.array(tvecs))

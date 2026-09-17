import open3d as o3d
import numpy as np
from scipy.io import loadmat

# Read depth map for image 0.
def read_depth(filepath="data/depth_0.mat"):
	data = loadmat(filepath)
	return np.asarray(data["depth_image"])

# Read keypoint matches between image 0 and image 1.
def read_matches(filepath="data/matches.mat"):
	data = loadmat(filepath)
	return np.asarray(data["pts0"]), np.asarray(data["pts1"])

# Read intrinsics for image 0.
def read_intrinsics(filepath="data/K_0.mat"):
	data = loadmat(filepath)
	return np.asarray(data["K"])

def read_img0(filepath="data/image_0.png"):
	image = np.asarray(o3d.io.read_image(filepath))
	return image

def read_img1(filepath="data/image_1.png"):
	image = np.asarray(o3d.io.read_image(filepath))
	return image

# Back-project keypoints to points in camera frame based on depth and intrinsics.
def get_img0_3d_keypoints(depth_map, keypoints, K):

	pixels = np.rint(keypoints).astype(int) # round the keypoints to the nearest int to be usable as indices

	depths = depth_map[pixels[:, 1], pixels[:, 0]]
	keypoints_hom = np.column_stack((
		keypoints,
		np.ones(keypoints.shape[0], dtype=float)
	)) # homogeneous keypoints (u, v, 1)

	# p_C = Z_C K^(-1) u
	return ((np.linalg.inv(K) @ keypoints_hom.T) * depths).T

# Back-project a whole image (colored or not) given the depth and intrinsics.
def backproject_img(image, depth, K):
	image = np.asarray(image)
	depth = np.asarray(depth, dtype=float)
	K = np.asarray(K, dtype=float)
	if image.ndim not in (2, 3):
		raise ValueError("image must have shape (height, width) or (height, width, channels)")
	if depth.ndim != 2 or image.shape[:2] != depth.shape:
		raise ValueError("image and depth must have matching height and width")
	if K.shape != (3, 3):
		raise ValueError("K must have shape (3, 3)")

	height, width = depth.shape
	u, v = np.meshgrid(np.arange(width), np.arange(height))
	pixels_hom = np.column_stack((u.ravel(), v.ravel(), np.ones(height * width)))
	rays = pixels_hom @ np.linalg.inv(K).T
	depth_values = depth.ravel()
	valid = np.isfinite(depth_values) & (depth_values > 0)
	points = rays[valid] * depth_values[valid, None]

	if image.ndim == 2:
		colors = np.repeat(image[..., None], 3, axis=2)
	else:
		colors = image[..., :3]
	colors = colors.reshape(-1, 3)[valid].astype(float)
	if np.issubdtype(image.dtype, np.integer):
		colors /= np.iinfo(image.dtype).max

	return points, colors

# Calibrate camera 1 based on know 3D points in camera 0's frame and keypoints in camera 1's pixels. Solve using DLT.
def calib_cam1(points0, keypoints1):

	points0_hom = np.column_stack((points0, np.ones(points0.shape[0]))) # homogeneous points (X_c0, Y_c0, Z_c0, 1)
	u, v = keypoints1[:, 0], keypoints1[:, 1]

	A = np.empty((2 * len(points0), 12), dtype=float) # A has shape (2N, 12) - 12=3x4 camera matrix P

	# even rows
	# X_i^T		0^T		-uX_i^T
	A[0::2] = np.column_stack((
		points0_hom,
		np.zeros((points0.shape[0], 4)),
		-u[:,None] * points0_hom
	))

	# odd rows
	# 0^T		X_i^T	-vX_i^T
	A[1::2] = np.column_stack((
		np.zeros((points0.shape[0], 4)),
		points0_hom,
		-v[:,None] * points0_hom
	))

	# solve for Ap=0
	U, S, Vh = np.linalg.svd(A)
	P = Vh[-1].reshape(3, 4) # the last right-singular vector. Is the eigenvector with the smallest eigenvalue.
	P /= np.linalg.norm(P)
	return P

# Get K[R|t] from the P matrix, using QR factorization.
def get_k_r_t_from_p(P):

	# get the leading 3x3 P block
	P0 = P[:3,:3]

	Q_, R_ = np.linalg.qr(np.linalg.inv(P0));

	# the rotation matrix is the inverse of the decomposed Q matrix
	R = np.linalg.inv(Q_)

	# the intrinsic matrix is the inverse of the decomposed R matrix
	K = np.linalg.inv(R_)

	# the translation vector is obtained by  multiplying the inverse of the intrinsic by the last column of P
	t = np.linalg.inv(K) @ P[:,-1]

	return K, R, t


# Visualize the point cloud interatively.
def render_pointcloud(pointcloud, colors):
	pointcloud = np.asarray(pointcloud, dtype=float)
	colors = np.asarray(colors, dtype=float)
	if colors.shape != (len(pointcloud), 3):
		raise ValueError("colors must have shape (n, 3)")

	cloud = o3d.geometry.PointCloud()
	cloud.points = o3d.utility.Vector3dVector(pointcloud)
	cloud.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
	o3d.visualization.draw_geometries([cloud], window_name="Point Cloud")
	return cloud

# Get the corresponding camera 1 pixels for camera 0 points. Needs camera 1's projective matrix P.
def get_cam1_pixels_for_cam0_points(points0, P_cam1):

	points0_hom = np.column_stack((points0, np.ones(len(points0))))	# expand the 3D points to homogeneous (X, Y, Z, 1)
	pixels_hom = points0_hom @ P_cam1.T		# get homogeneous pixels after projection in camera 1 lambda(u_1, v_1, 1)
	return pixels_hom[:, :2] / pixels_hom[:, 2, None] # divide by the labda scale to get back pixels

def get_img_colors(img, pixels):

	pixel_indices = np.rint(pixels).astype(int)

	# create a mask of valid pixel indices
	valid = (
		(pixel_indices[:, 0] >= 0)
		& (pixel_indices[:, 0] < img.shape[1])
		& (pixel_indices[:, 1] >= 0)
		& (pixel_indices[:, 1] < img.shape[0])
	)
	colors = np.zeros((pixels.shape[0], 3), dtype=float)
	colors[valid] = img[
		pixel_indices[valid, 1], pixel_indices[valid, 0], :3
	].astype(float) / 255.0
	return colors

# read images, depth map, keypoint matches and camera 0's intrinsics
img0 = read_img0()
img1 = read_img1()
depth_map = read_depth()
pts0, pts1 = read_matches()
K = read_intrinsics()

# back-project camera 0's keypoints
keypoints_3d = get_img0_3d_keypoints(depth_map, pts0, K)

# calibrate camera 1 relative to camera 0's points
P1 = calib_cam1(keypoints_3d, pts1)
print(f"P={P1}")

# recover K[R|t]
K1, R1, t1 = get_k_r_t_from_p(P1)
print(f"K1={K1}")
print(f"R1={R1}")
print(f"t1={t1}")
print(f"K[R|t]={K1 @ np.column_stack((R1,t1))}")


pointcloud0, _ = backproject_img(img0, depth_map, K) # back-project whole image 0
cam1_pixels = get_cam1_pixels_for_cam0_points(pointcloud0, P1) # get camera 0's points projected into camera 1
cam1_colors = get_img_colors(img1, cam1_pixels) # get the colors from the projected camera 1 image

render_pointcloud(pointcloud0, cam1_colors)


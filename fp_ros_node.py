#!/usr/bin/env python

import argparse
import logging
import os
import time

import cv2
import numpy as np
import nvdiffrast.torch as dr
import open3d as o3d
import pyrender
import rospy
import trimesh
from cv_bridge import CvBridge, CvBridgeError
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Header

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import (
    depth2xyzmap,
    draw_posed_3d_box,
    draw_xyz_axis,
    set_logging_format,
    set_seed,
    toOpen3dCloud,
)


def compare_masks(mask1, mask2, threshold=0.2):
    # Calculate the intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # Handle the case where union is zero to avoid division by zero
    if union == 0:
        if intersection == 0:
            return 1.0, True
        else:
            return 0.0, False

    # Compute Intersection over Union (IoU)
    iou = intersection / union

    # Check if IoU meets the threshold
    return iou, iou >= threshold


# Path to the mesh file


def render_depth_and_mask(trimesh_obj, T_C_O, K, image_width=640, image_height=480):
    """
    Render a depth image and mask image given a trimesh object and an object pose.

    Args:
        trimesh_obj: Trimesh object loaded using pyrender.Mesh.from_trimesh().
        T_C_O: 4x4 numpy array representing the object pose relative to the camera.
        K: 3x3 numpy array representing the camera intrinsic matrix.
        image_width: Width of the rendered image (default: 640).
        image_height: Height of the rendered image (default: 480).

    Returns:
        depth_image: 2D numpy array representing the depth image.
        mask: 2D numpy array representing the mask image.
    """
    # Define the rotation matrix for the camera
    R_C2_C = R.from_euler("x", 180, degrees=True).as_matrix()
    T_C2_C = np.eye(4)
    T_C2_C[:3, :3] = R_C2_C
    T_C2_O = T_C2_C @ T_C_O

    # Create a scene
    scene = pyrender.Scene()

    # Add the mesh to the scene
    mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
    scene.add(mesh, pose=T_C2_O)

    # Set up the camera intrinsics (f_x, f_y, c_x, c_y)
    f_x = K[0, 0]
    f_y = K[1, 1]
    c_x = K[0, 2]
    c_y = K[1, 2]

    camera = pyrender.IntrinsicsCamera(fx=f_x, fy=f_y, cx=c_x, cy=c_y)
    scene.add(camera)

    # Create an offscreen renderer
    renderer = pyrender.OffscreenRenderer(image_width, image_height)

    # Render the depth image and compute the mask
    depth_image = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
    mask = depth_image > 0

    # Return the depth image and mask
    return depth_image, mask


def render_depth_and_mask_cache(
    trimesh_obj, T_C_O, K, image_width=640, image_height=480
):
    """
    Render a depth image and mask image given a trimesh object and an object pose.

    Args:
        trimesh_obj: Trimesh object loaded using pyrender.Mesh.from_trimesh().
        T_C_O: 4x4 numpy array representing the object pose relative to the camera.
        K: 3x3 numpy array representing the camera intrinsic matrix.
        image_width: Width of the rendered image (default: 640).
        image_height: Height of the rendered image (default: 480).

    Returns:
        depth_image: 2D numpy array representing the depth image.
        mask: 2D numpy array representing the mask image.
    """
    import sys

    def printerr(x):
        print(x, file=sys.stderr)

    if not hasattr(render_depth_and_mask_cache, "first"):
        printerr("~" * 100)
        printerr("FIRST")
        printerr("~" * 100)

        render_depth_and_mask_cache.first = False

        # Define the rotation matrix for the camera
        R_C2_C = R.from_euler("x", 180, degrees=True).as_matrix()
        T_C2_C = np.eye(4)
        T_C2_C[:3, :3] = R_C2_C
        T_C2_O = T_C2_C @ T_C_O

        # Create a scene
        scene = pyrender.Scene()

        # Add the mesh to the scene
        mesh = pyrender.Mesh.from_trimesh(trimesh_obj)
        mesh_node = scene.add(mesh, pose=T_C2_O)

        # Set up the camera intrinsics (f_x, f_y, c_x, c_y)
        f_x = K[0, 0]
        f_y = K[1, 1]
        c_x = K[0, 2]
        c_y = K[1, 2]

        camera = pyrender.IntrinsicsCamera(fx=f_x, fy=f_y, cx=c_x, cy=c_y)
        scene.add(camera)

        # Create an offscreen renderer
        renderer = pyrender.OffscreenRenderer(image_width, image_height)

        # Render the depth image and compute the mask
        depth_image = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        mask = depth_image > 0

        render_depth_and_mask_cache.T_C2_C = T_C2_C
        render_depth_and_mask_cache.scene = scene
        render_depth_and_mask_cache.mesh = mesh
        render_depth_and_mask_cache.mesh_node = mesh_node
        render_depth_and_mask_cache.camera = camera
        render_depth_and_mask_cache.renderer = renderer

        # Return the depth image and mask
        return depth_image, mask
    else:
        printerr("~" * 100)
        printerr("NOT FIRST")
        printerr("~" * 100)
        T_C2_C = render_depth_and_mask_cache.T_C2_C
        mesh_node = render_depth_and_mask_cache.mesh_node
        scene = render_depth_and_mask_cache.scene
        renderer = render_depth_and_mask_cache.renderer

        # Update the pose of the mesh node to move the mesh
        # For example, moving the mesh by applying a new transformation matrix
        T_C2_O = T_C2_C @ T_C_O
        scene.set_pose(mesh_node, pose=T_C2_O)

        # Re-render the scene with the new mesh pose
        depth_image = renderer.render(scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        mask = depth_image > 0
        return depth_image, mask


class FoundationPoseROS:
    def __init__(self, args):
        set_logging_format()
        set_seed(0)

        # Variables for storing the latest images
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_mask = None
        self.frame_count = 0

        self.est_refine_iter = args.est_refine_iter
        self.track_refine_iter = args.track_refine_iter
        self.debug = args.debug
        self.debug_dir = args.debug_dir
        os.makedirs(self.debug_dir, exist_ok=True)
        os.makedirs(f"{self.debug_dir}/track_vis", exist_ok=True)
        os.makedirs(f"{self.debug_dir}/ob_in_cam", exist_ok=True)

        rospy.init_node("fp_node")
        self.bridge = CvBridge()

        # Load object mesh
        self.object_mesh = trimesh.load(args.mesh_file)
        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.object_mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        # FOUNDATION POSE initialization
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.FPModel = FoundationPose(
            model_pts=self.object_mesh.vertices,
            model_normals=self.object_mesh.vertex_normals,
            mesh=self.object_mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            debug_dir=self.debug_dir,
            debug=self.debug,
            glctx=self.glctx,
        )
        logging.info("Estimator initialization done")

        # Camera parameters
        code_dir = os.path.dirname(os.path.realpath(__file__))
        cam_K_file = f"{code_dir}/demo_data/blueblock/blueblock_occ_slide/cam_K.txt"
        rospy.loginfo(f"cam_K_file = {cam_K_file}")
        self.cam_K = np.loadtxt(cam_K_file).reshape(3, 3)

        # Subscribers for RGB, depth, and mask images
        self.rgb_sub = rospy.Subscriber(
            "/camera/color/image_raw", ROSImage, self.rgb_callback, queue_size=1
        )
        self.depth_sub = rospy.Subscriber(
            "/camera/aligned_depth_to_color/image_raw",
            ROSImage,
            self.depth_callback,
            queue_size=1,
        )
        self.mask_sub = rospy.Subscriber(
            "/sam2_mask", ROSImage, self.mask_callback, queue_size=1
        )

        # Publisher for the object pose
        self.pose_pub = rospy.Publisher("/object_pose", ROSImage, queue_size=10)
        self.predicted_mask_pub = rospy.Publisher("/fp_mask", ROSImage, queue_size=10)

    def update_images(self, predicted_depth, predicted_mask):
        self.ax_depth.set_data(predicted_depth)
        self.ax_mask.set_data(predicted_mask)
        self.fig.canvas.flush_events()  # Update the canvas immediately

    def rgb_callback(self, data):
        try:
            # self.latest_rgb = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.latest_rgb = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert RGB image: {e}")

    def depth_callback(self, data):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(data, "64FC1")
            # self.latest_depth = self.bridge.imgmsg_to_cv2(data, "8UC1")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert depth image: {e}")

    def mask_callback(self, data):
        try:
            self.latest_mask = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert mask image: {e}")

    def run(self):
        while not rospy.is_shutdown():
            self.process_images()

    def process_images(self):
        if (
            self.latest_rgb is None
            or self.latest_depth is None
            or self.latest_mask is None
        ):
            rospy.logwarn(
                "Missing one of the required images (RGB, depth, mask). Waiting..."
            )
            return

        logging.info(f"Processing frame: {self.frame_count}")
        rgb = self.process_rgb(self.latest_rgb)
        depth = self.process_depth(self.latest_depth)
        mask = self.process_mask(self.latest_mask)
        rospy.loginfo(f"rgb: {rgb.shape}, {rgb.dtype}, {np.max(rgb)}, {np.min(rgb)}")
        rospy.loginfo(
            f"depth: {depth.shape}, {depth.dtype}, {np.max(depth)}, {np.min(depth)}, {np.mean(depth)}, {np.median(depth)}"
        )
        rospy.loginfo(
            f"mask: {mask.shape}, {mask.dtype}, {np.max(mask)}, {np.min(mask)}"
        )

        # Estimation and tracking
        if self.frame_count == 0:  # Slow 1Hz mask generation + estimation
            pose = self.FPModel.register(
                K=self.cam_K,
                rgb=rgb,
                depth=depth,
                ob_mask=mask,
                iteration=self.est_refine_iter,
            )
            logging.info("First frame estimation done")
            rospy.logerr(f"pose.shape = {pose.shape}")
            rospy.logerr(f"pose = {pose}")

            if self.debug >= 3:
                m = self.object_mesh.copy()
                m.apply_transform(pose)
                m.export(f"{self.debug_dir}/model_tf.obj")
                xyz_map = depth2xyzmap(depth, self.cam_K)
                valid = depth >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
                o3d.io.write_point_cloud(f"{self.debug_dir}/scene_complete.ply", pcd)

            t0 = time.time()
            predicted_depth, predicted_mask = render_depth_and_mask_cache(
                self.object_mesh, pose, self.cam_K, image_width=640, image_height=480
            )
            rospy.logerr(f"time for pred mask is = {(time.time() - t0)*1000} ms")

        else:  # Fast 30Hz tracking
            pose = self.FPModel.track_one(
                rgb=rgb, depth=depth, K=self.cam_K, iteration=self.track_refine_iter
            )

            rospy.logerr(f"pose.shape = {pose.shape}")
            rospy.logerr(f"pose = {pose}")
            t0 = time.time()
            predicted_depth, predicted_mask = render_depth_and_mask_cache(
                self.object_mesh, pose, self.cam_K, image_width=640, image_height=480
            )
            rospy.logerr(f"time for pred mask is = {(time.time() - t0)*1000} ms")

            iou, is_match = compare_masks(mask, predicted_mask, threshold=0.2)

            is_match = True  # HACK
            rospy.logerr("=" * 100)
            rospy.logerr(f"IoU: {iou}")
            if is_match:
                rospy.logerr("Masks match within the threshold.")
            else:
                rospy.logerr("Masks do not match within the threshold.")

                # pose = self.FPModel.register(K=self.cam_K, rgb=rgb, depth=depth, ob_mask=mask, iteration=self.est_refine_iter)
                pose = self.FPModel.register(
                    K=self.cam_K,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=self.process_mask(self.latest_mask),
                    iteration=self.est_refine_iter,
                )

                logging.info("First frame estimation done")
                rospy.logerr(f"pose.shape = {pose.shape}")
                rospy.logerr(f"pose = {pose}")

                t0 = time.time()
                predicted_depth, predicted_mask = render_depth_and_mask_cache(
                    self.object_mesh,
                    pose,
                    self.cam_K,
                    image_width=640,
                    image_height=480,
                )
                rospy.logerr(f"time for pred mask is = {(time.time() - t0)*1000} ms")

                if self.debug >= 3:
                    m = self.object_mesh.copy()
                    m.apply_transform(pose)
                    m.export(f"{self.debug_dir}/model_tf.obj")
                    xyz_map = depth2xyzmap(depth, self.cam_K)
                    valid = depth >= 0.001
                    pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
                    o3d.io.write_point_cloud(
                        f"{self.debug_dir}/scene_complete.ply", pcd
                    )

            rospy.logerr("=" * 100)

        # Convert OpenCV image (mask) to ROS Image message
        mask_msg = self.bridge.cv2_to_imgmsg(
            predicted_mask.astype(np.uint8) * 255, encoding="8UC1"
        )
        mask_msg.header = Header(stamp=rospy.Time.now())

        # Publish the mask to the /sam2_mask topic
        self.predicted_mask_pub.publish(mask_msg)
        rospy.loginfo("Predicted mask published to /fp_mask")

        # Publish pose
        self.publish_pose(pose)

        if self.debug >= 1:
            center_pose = pose @ np.linalg.inv(self.to_origin)
            vis_img = rgb.copy()

            # Must be BGR for cv2
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

            vis_img = draw_posed_3d_box(
                self.cam_K, img=vis_img, ob_in_cam=center_pose, bbox=self.bbox
            )
            vis_img = draw_xyz_axis(
                vis_img,
                ob_in_cam=center_pose,
                scale=0.1,
                K=self.cam_K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )

            cv2.imshow("Pose Visualization", vis_img)
            cv2.waitKey(1)

        self.frame_count += 1

    def process_rgb(self, rgb):
        rospy.logdebug(f"rgb.shape = {rgb.shape}")
        rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_NEAREST)
        rospy.logdebug(f"AFTER rgb.shape = {rgb.shape}")
        return rgb

    def process_depth(self, depth):
        rospy.logdebug(f"depth.shape = {depth.shape}")
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)
        rospy.logdebug(f"AFTER depth.shape = {depth.shape}")
        depth = depth / 1000

        depth[depth < 0.1] = 0
        depth[depth > 4] = 0

        return depth

    def process_mask(self, mask):
        rospy.logdebug(f"mask.shape = {mask.shape}")
        mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST).astype(
            bool
        )
        rospy.logdebug(f"AFTER mask.shape = {mask.shape}")
        return mask

    def publish_pose(self, pose):
        # Convert the pose matrix into a ROS message
        pose_msg = ROSImage()
        pose_msg.header = Header(stamp=rospy.Time.now())
        pose_msg.encoding = "rgb8"
        pose_msg.height = 1
        pose_msg.width = 4
        pose_msg.data = pose.flatten().tolist()

        # Publish the pose
        self.pose_pub.publish(pose_msg)
        rospy.logdebug("Pose published to /object_pose")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/kiri_meshes/snackbox/3DModel.obj')
    # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/kiri_meshes/blueblock/3DModel.obj')
    parser.add_argument(
        "--mesh_file", type=str, default=f"{code_dir}/kiri_meshes/cup_ycbv/textured.obj"
    )
    parser.add_argument("--est_refine_iter", type=int, default=5)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=3)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    args = parser.parse_args()

    node = FoundationPoseROS(args)
    node.run()

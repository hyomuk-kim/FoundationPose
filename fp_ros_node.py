#!/usr/bin/env python3
"""
FoundationPose ROS2 node.
Subscribes to RGB-D camera images and SAM2 mask,
runs FoundationPose estimation/tracking,
and publishes the object pose as PoseStamped on /object_pose.
"""

import os
import time

import cv2
import numpy as np
import nvdiffrast.torch as dr
import rclpy
from rclpy.node import Node
import trimesh
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Int32

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from fp_ros_utils import get_mesh_file
from Utils import (
    draw_posed_3d_box,
    draw_xyz_axis,
    set_logging_format,
    set_seed,
)


class FoundationPoseROS2(Node):

    def __init__(self):
        super().__init__("fp_node")

        set_logging_format()
        set_seed(0)

        # State variables
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_cam_K = None
        self.latest_mask = None
        self.is_object_registered = False
        self.first = True

        # Refinement iterations
        self.first_est_refine_iter = 5  # Higher quality for first registration
        self.est_refine_iter = 1  # Fast re-init when reset triggered
        self.track_refine_iter = 2  # Per-frame tracking

        # FoundationPose library's internal debug level (passed to the model below).
        # Only >= 2 does anything: it dumps point clouds / refiner-vis images to disk,
        # which is slow. Keep at 0 for normal runs; bump manually when deep-debugging.
        code_dir = os.path.dirname(os.path.realpath(__file__))
        self.debug = 0
        self.debug_dir = f"{code_dir}/debug"

        # Our node's own real-time visualization (cv2 window). Separate from the
        # library debug above. Toggle at launch with -p visualize:=true.
        self.declare_parameter("visualize", True)
        self.visualize = self.get_parameter(
            "visualize").get_parameter_value().bool_value

        # Processing lock to prevent overlapping timer calls
        self.is_processing = False

        self.bridge = CvBridge()

        # Load object mesh
        mesh_file = get_mesh_file(self)
        self.object_mesh = trimesh.load(mesh_file)
        self.object_mesh.vertices *= 0.001  # Convert mesh from mm to meters
        self.to_origin, extents = trimesh.bounds.oriented_bounds(
            self.object_mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        # FoundationPose model init
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
        self.get_logger().info("FoundationPose model initialized")

        # Camera topic selection via ROS2 parameter
        self.declare_parameter("camera", "realsense")
        camera = self.get_parameter("camera").get_parameter_value().string_value
        self.get_logger().info(f"Using camera: {camera}")

        if camera == "zed":
            rgb_topic = "/zed/zed_node/rgb/image_rect_color"
            depth_topic = "/zed/zed_node/depth/depth_registered"
            cam_info_topic = "/zed/zed_node/rgb/camera_info"
        elif camera == "realsense":
            rgb_topic = "/camera/color/image_raw"
            depth_topic = "/camera/aligned_depth_to_color/image_raw"
            cam_info_topic = "/camera/color/camera_info"
        else:
            raise ValueError(f"Unknown camera: {camera}")

        # Subscribers
        self.create_subscription(ROSImage, rgb_topic, self.rgb_callback, 1)
        self.create_subscription(ROSImage, depth_topic, self.depth_callback, 1)
        self.create_subscription(ROSImage, "/sam2_mask", self.mask_callback, 1)
        self.create_subscription(CameraInfo, cam_info_topic,
                                 self.cam_K_callback, 1)
        self.create_subscription(Int32, "/fp_reset", self.reset_callback, 1)

        # Publisher: PoseStamped instead of Pose (adds timestamp)
        self.pose_pub = self.create_publisher(PoseStamped, "/object_pose", 1)

        # Timer-driven main loop (runs as fast as GPU allows)
        self.timer = self.create_timer(0.01, self.run_once)
        self.get_logger().info("FoundationPose ROS2 node ready")

    # ---------- callbacks ----------

    def rgb_callback(self, data):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            self.get_logger().error(f"RGB conversion failed: {e}")

    def depth_callback(self, data):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(data, "64FC1")
        except CvBridgeError as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def mask_callback(self, data):
        try:
            self.latest_mask = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            self.get_logger().error(f"Mask conversion failed: {e}")

    def cam_K_callback(self, data: CameraInfo):
        self.latest_cam_K = np.array(data.k).reshape(3, 3)

    def reset_callback(self, data: Int32):
        if data.data > 0:
            self.get_logger().info("Reset triggered — re-registering object")
            self.is_object_registered = False
        else:
            self.get_logger().info(
                "Reset message received with data <= 0, ignoring")

    # ---------- main loop ----------

    def run_once(self):
        """Called by timer. Runs one registration or tracking step."""
        if self.is_processing:
            return

        if any(x is None for x in [
                self.latest_rgb, self.latest_depth, self.latest_mask,
                self.latest_cam_K
        ]):
            self.get_logger().warn(
                "Waiting for RGB, depth, mask, and camera_info...",
                throttle_duration_sec=2.0)
            return

        self.is_processing = True
        try:
            if not self.is_object_registered:
                self._register()
            else:
                self._track()
        finally:
            self.is_processing = False

    def _register(self):
        """Initial pose estimation using SAM2 mask."""
        self.get_logger().info("Running registration...")
        rgb = self.process_rgb(self.latest_rgb)
        depth = self.process_depth(self.latest_depth)
        mask = self.process_mask(self.latest_mask)
        cam_K = self.latest_cam_K.copy()

        t0 = time.time()
        pose = self.FPModel.register(
            K=cam_K,
            rgb=rgb,
            depth=depth,
            ob_mask=mask,
            iteration=self.first_est_refine_iter
            if self.first else self.est_refine_iter,
        )
        elapsed_ms = (time.time() - t0) * 1000
        self.get_logger().info(
            f"Registration done in {elapsed_ms:.1f} ms, pose:\n{pose}")
        assert pose.shape == (4, 4), f"Unexpected pose shape: {pose.shape}"
        self.is_object_registered = True
        self.first = False

    def _track(self):
        """Frame-to-frame tracking."""
        rgb = self.process_rgb(self.latest_rgb)
        depth = self.process_depth(self.latest_depth)
        cam_K = self.latest_cam_K.copy()

        t0 = time.time()
        pose = self.FPModel.track_one(rgb=rgb,
                                      depth=depth,
                                      K=cam_K,
                                      iteration=self.track_refine_iter)
        elapsed_ms = (time.time() - t0) * 1000
        self.get_logger().info(f"Tracking done in {elapsed_ms:.1f} ms")

        self.publish_pose(pose)

        if self.visualize:
            center_pose = pose @ np.linalg.inv(self.to_origin)
            vis_img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
            vis_img = draw_posed_3d_box(cam_K,
                                        img=vis_img,
                                        ob_in_cam=center_pose,
                                        bbox=self.bbox)
            vis_img = draw_xyz_axis(vis_img,
                                    ob_in_cam=center_pose,
                                    scale=0.1,
                                    K=cam_K,
                                    thickness=3,
                                    transparency=0,
                                    is_input_rgb=True)
            cv2.imshow("Pose Visualization", vis_img)
            cv2.waitKey(1)

    # ---------- helpers ----------

    def publish_pose(self, pose: np.ndarray):
        assert pose.shape == (4, 4), f"Unexpected pose shape: {pose.shape}"
        trans = pose[:3, 3]
        quat_xyzw = R.from_matrix(pose[:3, :3]).as_quat()

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"  # adjust if needed
        msg.pose.position.x = float(trans[0])
        msg.pose.position.y = float(trans[1])
        msg.pose.position.z = float(trans[2])
        msg.pose.orientation.x = float(quat_xyzw[0])
        msg.pose.orientation.y = float(quat_xyzw[1])
        msg.pose.orientation.z = float(quat_xyzw[2])
        msg.pose.orientation.w = float(quat_xyzw[3])
        self.pose_pub.publish(msg)

    def process_rgb(self, rgb):
        return rgb

    def process_depth(self, depth):
        depth = depth.copy()
        depth[np.isnan(depth)] = 0
        depth[np.isinf(depth)] = 0
        if depth.max() > 100:  # mm → m
            depth = depth / 1000.0
        depth[depth < 0.1] = 0
        depth[depth > 4.0] = 0
        return depth

    def process_mask(self, mask):
        return mask.astype(bool)


def main(args=None):
    rclpy.init(args=args)
    node = FoundationPoseROS2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

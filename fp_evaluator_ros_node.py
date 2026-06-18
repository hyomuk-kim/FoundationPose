#!/usr/bin/env python3
"""
FoundationPose Evaluator ROS2 node.
Compares the rendered mask from the estimated pose against the SAM2 mask
to compute IoU. Publishes a reset signal (/fp_reset) when tracking is lost.
"""

import time
import numpy as np
import rclpy
import cv2
import trimesh

from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge, CvBridgeError

from rclpy.node import Node
from std_msgs.msg import Int32, Float32
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ROSImage
from geometry_msgs.msg import PoseStamped

from fp_ros_utils import get_mesh_file
from evaluator import render_depth_and_mask_cache, compare_masks
from Utils import draw_posed_3d_box, draw_xyz_axis


class FoundationPoseEvaluatorROS2(Node):

    def __init__(self):
        super().__init__("fp_evaluator_node")

        # State
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_mask = None
        self.latest_cam_K = None
        self.latest_pose = None
        self.frame_count = 0

        self.bridge = CvBridge()

        # Load object mesh
        mesh_file = get_mesh_file(self)
        self.object_mesh = trimesh.load(mesh_file)
        self.to_origin, extents = trimesh.bounds.oriented_bounds(
            self.object_mesh)
        self.bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

        # Camera topic selection
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
        # PoseStamped instead of Pose (matches fp_ros_node.py ROS2 publisher)
        self.create_subscription(PoseStamped, "/object_pose",
                                 self.pose_callback, 1)

        # Publishers
        self.iou_pub = self.create_publisher(Float32, "/iou", 1)
        self.reset_pub = self.create_publisher(Int32, "/fp_reset", 1)
        self.predicted_mask_pub = self.create_publisher(ROSImage, "/fp_mask", 1)

        # Reset logic state
        RATE_HZ = 10
        self.RESET_COOLDOWN_SEC = 1.0
        self.INVALID_THRESHOLD_SEC = 1.0
        self.invalid_counter_threshold = int(self.INVALID_THRESHOLD_SEC *
                                             RATE_HZ)
        self.invalid_counter = 0
        self.last_reset_time = self.get_clock().now()

        # Visualize flag (set via ROS2 parameter)
        self.declare_parameter("visualize", True)
        self.visualize = self.get_parameter(
            "visualize").get_parameter_value().bool_value

        # Optional depth-based check (off by default).
        # Mask IoU only compares the 2D silhouette, so an object shifted along the
        # camera (z) axis can still pass. When enabled, we also compare the rendered
        # depth against the measured depth inside the mask to catch depth-only errors.
        self.declare_parameter("use_depth_check", False)
        self.use_depth_check = self.get_parameter(
            "use_depth_check").get_parameter_value().bool_value
        # Median depth difference (meters) above which the pose is considered a mismatch.
        self.declare_parameter("depth_check_threshold", 0.03)
        self.depth_check_threshold = self.get_parameter(
            "depth_check_threshold").get_parameter_value().double_value

        if self.use_depth_check:
            self.get_logger().info(
                f"Depth check enabled (threshold = {self.depth_check_threshold} m)"
            )

        # Timer-driven main loop at 10 Hz
        self.timer = self.create_timer(1.0 / RATE_HZ, self.run_once)
        self.get_logger().info("FoundationPose Evaluator ROS2 node ready")

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

    def pose_callback(self, data: PoseStamped):
        # PoseStamped → 4x4 matrix
        xyz = np.array([
            data.pose.position.x,
            data.pose.position.y,
            data.pose.position.z,
        ])
        quat_xyzw = np.array([
            data.pose.orientation.x,
            data.pose.orientation.y,
            data.pose.orientation.z,
            data.pose.orientation.w,
        ])
        pose = np.eye(4)
        pose[:3, 3] = xyz
        pose[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        self.latest_pose = pose

    # ---------- main loop ----------

    def run_once(self):
        required = [
            self.latest_rgb, self.latest_mask, self.latest_cam_K,
            self.latest_pose
        ]
        if self.use_depth_check:
            required.append(self.latest_depth)

        if any(x is None for x in required):
            self.get_logger().warn(
                "Waiting for required inputs (RGB, mask, cam_K, pose" +
                (", depth" if self.use_depth_check else "") + ")...",
                throttle_duration_sec=2.0)
            return

        self.frame_count += 1
        start = self.get_clock().now()

        rgb = self.process_rgb(self.latest_rgb)
        mask = self.process_mask(self.latest_mask)
        cam_K = self.latest_cam_K.copy()
        pose = self.latest_pose.copy()

        height, width = mask.shape[:2]
        t0 = time.time()
        predicted_depth, predicted_mask = render_depth_and_mask_cache(
            trimesh_obj=self.object_mesh,
            T_C_O=pose,
            K=cam_K,
            image_width=width,
            image_height=height,
        )
        self.get_logger().info(
            f"Render time: {(time.time() - t0) * 1000:.1f} ms")

        iou, is_match = compare_masks(mask, predicted_mask, threshold=0.2)
        self.get_logger().info(f"IoU: {iou:.3f}")
        self.iou_pub.publish(Float32(data=float(iou)))

        # Optional depth check: even if the mask matches, a wrong z-distance
        # should count as a mismatch. Both checks must pass to be a match.
        if self.use_depth_check:
            depth_ok = self._check_depth(
                measured_depth=self.process_depth(self.latest_depth),
                predicted_depth=predicted_depth,
                predicted_mask=predicted_mask,
            )
            is_match = is_match and depth_ok

        # Reset logic
        reset_msg = Int32(data=0)
        now = self.get_clock().now()
        elapsed_since_reset = (now - self.last_reset_time).nanoseconds / 1e9

        if is_match:
            self.invalid_counter = 0
        else:
            if elapsed_since_reset < self.RESET_COOLDOWN_SEC:
                self.get_logger().info(
                    f"In reset cooldown ({elapsed_since_reset:.2f}s / {self.RESET_COOLDOWN_SEC}s)"
                )
                self.invalid_counter = 0
            else:
                self.invalid_counter += 1
                self.get_logger().warn(
                    f"Mask mismatch: invalid_counter = {self.invalid_counter} / {self.invalid_counter_threshold}"
                )
                if self.invalid_counter >= self.invalid_counter_threshold:
                    self.get_logger().warn("Triggering reset")
                    reset_msg.data = 1
                    self.invalid_counter = 0
                    self.last_reset_time = now

        self.reset_pub.publish(reset_msg)

        # Publish predicted mask
        mask_msg = self.bridge.cv2_to_imgmsg(predicted_mask.astype(np.uint8) *
                                             255,
                                             encoding="8UC1")
        mask_msg.header.stamp = self.get_clock().now().to_msg()
        self.predicted_mask_pub.publish(mask_msg)

        # Visualization (enabled via --ros-args -p visualize:=true)
        if self.visualize:
            center_pose = pose @ np.linalg.inv(self.to_origin)
            vis_img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
            vis_img = draw_posed_3d_box(cam_K,
                                        img=vis_img,
                                        ob_in_cam=center_pose,
                                        bbox=self.bbox)
            vis_img = draw_xyz_axis(
                vis_img,
                ob_in_cam=center_pose,
                scale=0.1,
                K=cam_K,
                thickness=3,
                transparency=0,
                is_input_rgb=True,
            )
            cv2.imshow("Evaluator: Pose Visualization", vis_img)
            cv2.waitKey(1)

        # Log actual processing rate
        elapsed_sec = (self.get_clock().now() - start).nanoseconds / 1e9
        if elapsed_sec > 0:
            self.get_logger().info(
                f"Frame {self.frame_count}: {1.0 / elapsed_sec:.1f} Hz")

    # ---------- helpers ----------

    def process_rgb(self, rgb):
        return rgb

    def process_mask(self, mask):
        return mask.astype(bool)

    def process_depth(self, depth):
        depth = depth.copy()
        depth[np.isnan(depth)] = 0
        depth[np.isinf(depth)] = 0
        if depth.max() > 100:  # mm → m
            depth = depth / 1000.0
        depth[depth < 0.1] = 0
        depth[depth > 4.0] = 0
        return depth

    def _check_depth(self, measured_depth, predicted_depth, predicted_mask):
        """
        Compare the rendered depth against the measured depth inside the mask.
        Returns True if the median depth difference is within threshold.
        Uses median (robust to depth-sensor noise and edge outliers).
        """
        # Only compare where both the rendered object and the measured depth are valid
        valid = ((predicted_mask > 0) & (predicted_depth > 0) &
                 (measured_depth > 0))
        n_valid = int(valid.sum())
        if n_valid < 50:
            # Too few overlapping points to judge; don't fail on depth alone.
            self.get_logger().warn(
                f"Depth check skipped: only {n_valid} valid pixels")
            return True

        diff = np.abs(predicted_depth[valid] - measured_depth[valid])
        median_diff = float(np.median(diff))
        depth_ok = median_diff <= self.depth_check_threshold

        self.get_logger().info(
            f"Depth median diff: {median_diff * 1000:.1f} mm "
            f"(threshold {self.depth_check_threshold * 1000:.0f} mm) -> "
            f"{'OK' if depth_ok else 'MISMATCH'}")

        return depth_ok


def main(args=None):
    rclpy.init(args=args)
    node = FoundationPoseEvaluatorROS2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

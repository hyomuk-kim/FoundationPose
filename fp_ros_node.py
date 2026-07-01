#!/usr/bin/env python3
"""
FoundationPose ROS2 node.
Subscribes to RGB-D camera images and SAM2 mask,
runs FoundationPose estimation/tracking,
and publishes the object pose as PoseStamped on /object_pose.
"""

import os
import time
import numpy as np
import cv2
import nvdiffrast.torch as dr
import rclpy
import trimesh

from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Int32
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ROSImage

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
        self.latest_mask_stamp = None  # arrival time of latest mask (staleness)
        self.is_object_registered = False
        self.first = True

        # Constant-velocity SE(3) prior: seed each track step with an
        # extrapolation of the last two poses (pose_last is ob_in_cam).
        self.use_cv_prior = False
        self.pose_last_prev = None
        self.max_translation_step = 0.1  # reject >10cm/frame jumps as glitches

        # Mask-gated tracking: zero depth outside the SAM2 mask before tracking.
        self.use_mask_gating = False
        self.mask_gating_min_pixels = 100  # below this, mask is empty -> skip gating
        self.mask_gating_dilate_px = 15  # grow mask to absorb mask/object lag
        self.mask_gating_max_staleness_sec = 1.2  # skip gating if mask older (SAM2 ~1Hz)

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
        self.latest_vis_img = None

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
            # depth_topic = "/camera/depth/image_rect_raw"
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
            self.latest_mask_stamp = self.get_clock().now()
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
        self.pose_last_prev = None  # reset CV-prior history after re-registration

    def _track(self):
        """Frame-to-frame tracking."""
        rgb = self.process_rgb(self.latest_rgb)
        depth = self.process_depth(self.latest_depth)
        cam_K = self.latest_cam_K.copy()

        # Mask-gate depth + apply constant-velocity prior before tracking.
        depth = self.gate_depth_by_mask(depth)
        self.apply_cv_prior()

        t0 = time.time()
        pose = self.FPModel.track_one(rgb=rgb,
                                      depth=depth,
                                      K=cam_K,
                                      iteration=self.track_refine_iter)
        elapsed_ms = (time.time() - t0) * 1000
        self.get_logger().info(f"Tracking done in {elapsed_ms:.1f} ms")

        self.update_cv_prior_history()
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
            # cv2.imshow("Pose Visualization", vis_img)
            # cv2.waitKey(1)
            self.latest_vis_img = vis_img

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

    def gate_depth_by_mask(self, depth):
        """Zero depth outside the dilated SAM2 mask. Falls back to ungated depth
        if the mask is missing, empty, stale, or shape-mismatched."""
        # missing
        if not self.use_mask_gating or self.latest_mask is None:
            return depth

        # stale
        if self.latest_mask_stamp is not None:
            staleness = (self.get_clock().now() -
                         self.latest_mask_stamp).nanoseconds * 1e-9
            if staleness > self.mask_gating_max_staleness_sec:
                self.get_logger().warn(
                    f"Mask stale ({staleness * 1000:.0f} ms) — skipping gating",
                    throttle_duration_sec=2.0)
                return depth

        # shape-mismatched
        mask = self.latest_mask
        if mask.shape != depth.shape:
            self.get_logger().warn(
                f"Mask shape {mask.shape} != depth {depth.shape} — skipping gating",
                throttle_duration_sec=2.0)
            return depth

        # empty
        mask_bool = mask > 0
        if int(mask_bool.sum()) < self.mask_gating_min_pixels:
            self.get_logger().warn("Mask near-empty — skipping gating",
                                   throttle_duration_sec=2.0)
            return depth

        if self.mask_gating_dilate_px > 0:
            k = 2 * self.mask_gating_dilate_px + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask_bool = cv2.dilate(mask_bool.astype(np.uint8), kernel) > 0

        gated = depth.copy()
        gated[~mask_bool] = 0
        return gated

    # ---------- constant-velocity prior ----------

    def apply_cv_prior(self):
        """Seed FoundationPose.pose_last with a constant-velocity SE(3)
        extrapolation of the last two poses (ob_in_cam, so pred = delta @ curr)."""
        if not self.use_cv_prior:
            return

        pose_last = self.FPModel.pose_last
        if pose_last is None or self.pose_last_prev is None:
            return

        prev = self.pose_last_prev
        curr = pose_last.detach().cpu().numpy().reshape(4, 4)
        delta = curr @ np.linalg.inv(prev)
        pred = delta @ curr

        translation_step = np.linalg.norm(pred[:3, 3] - curr[:3, 3])
        if translation_step > self.max_translation_step:
            self.get_logger().warn(
                f"CV prior jump {translation_step*100:.1f} cm too large — skipping",
                throttle_duration_sec=2.0)
            return

        # Re-orthonormalize rotation against numerical drift.
        u, _, vt = np.linalg.svd(pred[:3, :3])
        pred[:3, :3] = u @ vt

        import torch
        self.FPModel.pose_last = torch.as_tensor(pred,
                                                 dtype=pose_last.dtype,
                                                 device=pose_last.device)

    def update_cv_prior_history(self):
        if not self.use_cv_prior:
            return
        pose_last = self.FPModel.pose_last
        if pose_last is None:
            return
        self.pose_last_prev = pose_last.detach().cpu().numpy().reshape(4, 4)


def main(args=None):
    import threading
    rclpy.init(args=args)
    node = FoundationPoseROS2()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            if node.visualize and node.latest_vis_img is not None:
                cv2.imshow("Pose Visualization", node.latest_vis_img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()

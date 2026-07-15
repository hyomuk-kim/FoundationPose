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
import torch
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
    nvdiffrast_render,
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
        self.use_cv_prior = True
        self.pose_last_prev = None
        self.max_translation_step = 0.1  # reject >10cm/frame jumps as glitches

        # Mask-gated tracking: zero depth outside the SAM2 mask before tracking.
        self.use_mask_gating = True
        self.mask_gating_min_pixels = 100  # below this, mask is empty -> skip gating
        self.mask_gating_dilate_px = 15  # grow mask to absorb mask/object lag
        self.mask_gating_max_staleness_sec = 1.2  # skip gating if mask older (SAM2 ~1Hz)

        # Auto-reset parameters for spatial drift detection
        self.use_auto_reset = True
        self.auto_reset_patience = 3  # Consecutive frames required to trigger reset
        self.drift_counter = 0
        self.max_center_dist_px = 30.0  # Max pixel distance between SAM2 and FP centers

        # Reset mode: centroid distance (translation-only) vs depth residual.
        # Depth residual compares the FP-rendered depth against observed depth
        # over co-visible, non-occluded pixels, so it also catches rotation
        # drift that leaves the centroid in place. Toggle at launch with
        # -p use_depth_residual_reset:=false to fall back to centroid mode.
        self.declare_parameter("use_depth_residual_reset", True)
        self.use_depth_residual_reset = self.get_parameter(
            "use_depth_residual_reset").get_parameter_value().bool_value
        self.declare_parameter("depth_residual_thresh_m", 0.015)
        self.depth_residual_thresh_m = self.get_parameter(
            "depth_residual_thresh_m").get_parameter_value().double_value
        # Min co-visible pixels for a valid residual (else hold, don't count).
        self.depth_residual_min_covisible_px = 200
        # Observed depth this much closer than rendered => an occluder in front.
        self.depth_residual_occlusion_margin_m = 0.03
        # If more than this fraction of co-visible pixels are occluded, the
        # object is heavily occluded => hold, never reset on this frame.
        self.depth_residual_heavy_occ_frac = 0.6

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

        # Debug visualization: render the FP mesh silhouette at the tracked pose
        # (the same geometry the refiner compares against internally) and publish
        # it alongside the SAM2 mask so drift can be inspected in rqt.
        # Toggle at launch with -p debug_viz:=false to save the per-frame render.
        self.declare_parameter("debug_viz", True)
        self.debug_viz = self.get_parameter(
            "debug_viz").get_parameter_value().bool_value

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

        # Debug-viz publishers: FP silhouette mask + RGB overlay (view in rqt).
        self.render_mask_pub = self.create_publisher(ROSImage,
                                                     "/fp_render_mask", 1)
        self.debug_overlay_pub = self.create_publisher(ROSImage,
                                                       "/fp_debug_overlay", 1)
        # Colorized |z_render - z_obs| heatmap over co-visible pixels.
        self.depth_residual_pub = self.create_publisher(ROSImage,
                                                        "/fp_depth_residual", 1)

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
        depth_obs = self.process_depth(self.latest_depth)  # ungated observation
        cam_K = self.latest_cam_K.copy()

        # Mask-gate depth + apply constant-velocity prior before tracking.
        depth = self.gate_depth_by_mask(depth_obs)
        self.apply_cv_prior()

        t0 = time.time()
        pose = self.FPModel.track_one(rgb=rgb,
                                      depth=depth,
                                      K=cam_K,
                                      iteration=self.track_refine_iter)
        elapsed_ms = (time.time() - t0) * 1000
        self.get_logger().info(f"Tracking done in {elapsed_ms:.1f} ms")

        # Render the FP depth once and reuse it for both debug viz and the
        # depth-residual reset (avoids rendering the mesh twice per frame).
        H, W = depth_obs.shape
        rendered_depth = None
        need_render = self.debug_viz or (self.use_auto_reset and
                                         self.use_depth_residual_reset)
        if need_render:
            rendered_depth = self.render_fp_depth(pose, cam_K, H, W)

        # Publish debug viz before the reset check so borderline / stuck frames
        # (which do NOT trigger a reset) are still inspectable in rqt.
        if self.debug_viz:
            self.publish_debug_viz(rgb, pose, cam_K, depth_obs, rendered_depth)

        if self.check_auto_reset(pose, cam_K, depth_obs, rendered_depth):
            return  # Abort publishing this frame and re-register next frame

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

    # ---------- auto-reset ----------

    def check_auto_reset(self,
                         pose: np.ndarray,
                         cam_K: np.ndarray,
                         observed_depth: np.ndarray = None,
                         rendered_depth=None) -> bool:
        """Dispatch to the configured reset criterion. Returns True if a reset
        was triggered. Depth-residual mode catches rotation drift that leaves
        the centroid in place; centroid mode is the translation-only fallback."""

        if not self.use_auto_reset:
            return False

        if self.use_depth_residual_reset:
            return self._check_reset_depth_residual(pose, cam_K, rendered_depth,
                                                    observed_depth)

        return self._check_reset_centroid(pose, cam_K)

    def _centroid_distance(self, pose: np.ndarray, cam_K: np.ndarray):
        """Pixel distance between the SAM2 mask centroid and the projected FP
        3D center. Returns None if there is no usable mask or the object is
        behind the camera."""

        if self.latest_mask is None:
            return None

        mask_bool = self.latest_mask > 0
        if int(mask_bool.sum()) < self.mask_gating_min_pixels:
            return None

        ys, xs = np.nonzero(mask_bool)
        mask_u, mask_v = float(np.mean(xs)), float(np.mean(ys))

        t = pose[:3, 3]
        if t[2] <= 0.01:  # behind the camera
            return None

        fp_u = (cam_K[0, 0] * t[0] / t[2]) + cam_K[0, 2]
        fp_v = (cam_K[1, 1] * t[1] / t[2]) + cam_K[1, 2]

        return float(np.hypot(mask_u - fp_u, mask_v - fp_v))

    def _check_reset_depth_residual(self, pose: np.ndarray, cam_K: np.ndarray,
                                    rendered_depth, observed_depth) -> bool:
        """Combined reset: depth residual (rotation-sensitive) OR centroid
        distance (gross divergence safety net). Resets when either votes drift
        for `auto_reset_patience` frames.

        The occlusion guard only silences the DEPTH vote (a briefly hidden but
        correct pose has high residual / is mostly occluded). The centroid net
        stays active so a lost object that drifted away -- which makes the depth
        residual read n/a -- is still caught. Occlusion is only trusted when the
        centroid agrees the object is roughly where FP thinks it is."""
        drift_vote = False
        reasons = []

        # (A) Centroid safety net -- catches gross divergence even when the
        # depth residual is n/a (object moved off the rendered silhouette).
        cdist = self._centroid_distance(pose, cam_K)
        centroid_far = cdist is not None and cdist > self.max_center_dist_px
        if centroid_far:
            drift_vote = True
            reasons.append(f"center {cdist:.0f}px")

        # (B) Depth residual -- rotation-sensitive, occlusion-gated.
        residual = None
        if rendered_depth is not None and observed_depth is not None:
            residual, n_covis, n_occ = self.compute_depth_residual(
                rendered_depth, observed_depth)
            if residual is not None:
                occ_frac = n_occ / max(1, n_covis)
                # Trust "heavy occlusion -> hold" only if the centroid agrees the
                # object is still in place; otherwise it is loss, not occlusion.
                heavy_occ = occ_frac > self.depth_residual_heavy_occ_frac
                if heavy_occ and not centroid_far:
                    pass  # genuinely occluded, do not vote from depth
                elif residual > self.depth_residual_thresh_m:
                    drift_vote = True
                    reasons.append(f"depth {residual*1000:.0f}mm")

        if drift_vote:
            self.drift_counter += 1
        else:
            self.drift_counter = max(0, self.drift_counter - 1)

        if self.drift_counter >= self.auto_reset_patience:
            self.get_logger().warn(
                f"Tracking lost ({', '.join(reasons)}). Auto-reset triggered.")
            self.is_object_registered = False
            self.drift_counter = 0
            return True

        return False

    def _check_reset_centroid(self, pose: np.ndarray,
                              cam_K: np.ndarray) -> bool:
        """
        Detects spatial drift by comparing the SAM2 mask centroid with the
        2D projection of the FoundationPose 3D center. Triggers a reset if lost.
        Returns True if a reset was triggered, False otherwise.
        """
        dist = self._centroid_distance(pose, cam_K)
        if dist is None:
            return False

        # Update drift counter
        if dist > self.max_center_dist_px:
            self.drift_counter += 1
        else:
            self.drift_counter = max(0, self.drift_counter - 1)

        # Trigger reset if patience is exceeded
        if self.drift_counter >= self.auto_reset_patience:
            self.get_logger().warn(
                f"Tracking lost (Center dist: {dist:.1f}px). Auto-reset triggered."
            )
            self.is_object_registered = False
            self.drift_counter = 0
            return True

        return False

    # ---------- FP mesh rendering + depth residual ----------

    def render_fp_depth(self, pose: np.ndarray, cam_K: np.ndarray, H: int,
                        W: int):
        """Render the FP mesh at the tracked pose and return its depth map
        (float32, meters; 0 = background). Uses the SAME centered mesh_tensors
        and pose_last convention the refiner compares against internally, so it
        reflects exactly what FoundationPose "sees" as the object. Returns None
        on failure so the tracking loop is never broken by rendering."""
        try:
            ob_in_cam = torch.as_tensor(pose.reshape(1, 4, 4),
                                        device="cuda",
                                        dtype=torch.float)
            _, depth_r, _ = nvdiffrast_render(
                K=cam_K,
                H=H,
                W=W,
                ob_in_cams=ob_in_cam,
                glctx=self.glctx,
                mesh_tensors=self.FPModel.mesh_tensors,
            )
            return depth_r[0].detach().cpu().numpy()  # (H, W), meters; 0 = bg
        except Exception as e:  # rendering must never break the tracking loop
            self.get_logger().warn(f"FP depth render failed: {e}",
                                   throttle_duration_sec=2.0)
            return None

    def compute_depth_residual(self, rendered_depth: np.ndarray,
                               observed_depth: np.ndarray):
        """Median |z_render - z_obs| over co-visible, non-occluded pixels.
        A pixel is co-visible when both rendered and observed depth are valid;
        an occluder is detected when the observation is clearly closer than the
        rendered object surface, and those pixels are excluded from the residual.
        Returns (residual_m or None, n_covisible, n_occluded)."""
        covis = (rendered_depth > 0) & (observed_depth > 0)
        n_covis = int(covis.sum())
        if n_covis < self.depth_residual_min_covisible_px:
            return None, n_covis, 0

        diff = observed_depth[covis] - rendered_depth[covis]
        occluded = diff < -self.depth_residual_occlusion_margin_m
        n_occ = int(occluded.sum())

        inliers = ~occluded
        if int(inliers.sum()) < self.depth_residual_min_covisible_px:
            return None, n_covis, n_occ

        residual = float(np.median(np.abs(diff[inliers])))
        return residual, n_covis, n_occ

    # ---------- debug visualization ----------

    def publish_debug_viz(self, rgb: np.ndarray, pose: np.ndarray,
                          cam_K: np.ndarray, observed_depth: np.ndarray,
                          rendered_depth):
        """Publish the FP silhouette (/fp_render_mask), an RGB overlay
        (/fp_debug_overlay) comparing the FP silhouette (green) with the SAM2
        mask (red), and a colorized depth-residual heatmap (/fp_depth_residual).
        Overlays centroid distance, mask IoU, and depth residual for diagnosis."""
        if rendered_depth is None:
            return

        H, W = rendered_depth.shape
        fp_sil = (rendered_depth > 0).astype(np.uint8) * 255
        fp_bool = fp_sil > 0

        # /fp_render_mask — raw FP silhouette
        try:
            self.render_mask_pub.publish(
                self.bridge.cv2_to_imgmsg(fp_sil, encoding="mono8"))
        except CvBridgeError as e:
            self.get_logger().warn(f"render_mask publish failed: {e}",
                                   throttle_duration_sec=2.0)

        # /fp_debug_overlay — RGB with FP silhouette + SAM2 mask overlaid
        overlay = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)

        # Translucent green fill for the FP silhouette
        if fp_bool.any():
            tint = overlay.copy()
            tint[fp_bool] = (0, 255, 0)
            overlay = cv2.addWeighted(overlay, 0.75, tint, 0.25, 0)
            cnts, _ = cv2.findContours(fp_sil, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 2)

        # SAM2 mask outline in red + centroid distance / IoU text
        if self.latest_mask is not None and self.latest_mask.shape == (H, W):
            sam_bool = self.latest_mask > 0
            sam_u8 = sam_bool.astype(np.uint8) * 255
            cnts, _ = cv2.findContours(sam_u8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, cnts, -1, (0, 0, 255), 2)

            # FP projected 3D center (blue) — matches the centroid reset metric
            t = pose[:3, 3]
            if t[2] > 0.01:
                fp_u = int((cam_K[0, 0] * t[0] / t[2]) + cam_K[0, 2])
                fp_v = int((cam_K[1, 1] * t[1] / t[2]) + cam_K[1, 2])
                cv2.circle(overlay, (fp_u, fp_v), 5, (255, 0, 0), -1)

                if sam_bool.any():
                    ys, xs = np.nonzero(sam_bool)
                    sam_u, sam_v = int(np.mean(xs)), int(np.mean(ys))
                    cv2.circle(overlay, (sam_u, sam_v), 5, (0, 0, 255), -1)
                    cv2.line(overlay, (fp_u, fp_v), (sam_u, sam_v),
                             (255, 255, 0), 1)
                    dist = float(np.hypot(fp_u - sam_u, fp_v - sam_v))
                    inter = int(np.logical_and(fp_bool, sam_bool).sum())
                    union = int(np.logical_or(fp_bool, sam_bool).sum())
                    iou = inter / union if union > 0 else 0.0
                    cv2.putText(overlay,
                                f"center_dist={dist:.1f}px  IoU={iou:.2f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2, cv2.LINE_AA)

        # Depth residual text (independent of SAM2). Always print covis / occ%,
        # even on n/a, so the two failure causes are distinguishable:
        #   - low covis  : n_covis below the min (few co-visible pixels at all)
        #   - heavy excl : n_covis fine but most pixels excluded as "occluded"
        #                  (broad exclusion -> a pose error masquerading as occ)
        residual, n_covis, n_occ = self.compute_depth_residual(
            rendered_depth, observed_depth)
        occ_frac = n_occ / max(1, n_covis)
        if residual is not None:
            txt = (f"depth_res={residual*1000:.1f}mm  "
                   f"occ={occ_frac*100:.0f}%  covis={n_covis}")
        else:
            txt = (f"depth_res=n/a  covis={n_covis}  occ={occ_frac*100:.0f}%  "
                   f"(min covis={self.depth_residual_min_covisible_px})")
        cv2.putText(overlay, txt, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255), 2, cv2.LINE_AA)

        try:
            self.debug_overlay_pub.publish(
                self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8"))
        except CvBridgeError as e:
            self.get_logger().warn(f"debug_overlay publish failed: {e}",
                                   throttle_duration_sec=2.0)

        # /fp_depth_residual — colorized |z_render - z_obs| over co-visible px.
        # Non co-visible = black; excluded occluders = white. Display is capped
        # at 3x the reset threshold so the useful range is well spread.
        covis = (rendered_depth > 0) & (observed_depth > 0)
        diff = observed_depth - rendered_depth
        cap = max(self.depth_residual_thresh_m * 3.0, 1e-6)
        norm = np.clip(np.abs(diff) / cap, 0.0, 1.0)
        heat = np.zeros((H, W), np.uint8)
        heat[covis] = (norm[covis] * 255).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat_color[~covis] = (0, 0, 0)
        occluded = covis & (diff < -self.depth_residual_occlusion_margin_m)
        heat_color[occluded] = (255, 255, 255)
        try:
            self.depth_residual_pub.publish(
                self.bridge.cv2_to_imgmsg(heat_color, encoding="bgr8"))
        except CvBridgeError as e:
            self.get_logger().warn(f"depth_residual publish failed: {e}",
                                   throttle_duration_sec=2.0)


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

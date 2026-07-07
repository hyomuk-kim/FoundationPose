#!/usr/bin/env python3
"""
FoundationPose ROS2 node.
Subscribes to RGB-D camera images and SAM2 mask,
runs FoundationPose estimation/tracking,
and publishes the object pose as PoseStamped on /object_pose.
"""

import os
import sys
import time
import math
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
    set_logging_format,
    set_seed,
)


def _hann2d(sz: int) -> torch.Tensor:
    """Center-weighted mask multiplied into the score map: keeps detections
    near the previous position, suppresses far-away look-alike distractors."""
    w = 0.5 * (1 - torch.cos(
        (2 * math.pi / (sz + 1)) * torch.arange(1, sz + 1).float()))
    return w.reshape(1, 1, -1, 1) * w.reshape(1, 1, 1, -1)


class Tracker2D:
    """OSTrack (vitb_256_mae_ce_32x4_ep300) wrapper for per-frame bbox
    tracking, independent of FP's own pose estimate. Ported from
    lib/test/tracker/{ostrack.py,data_utils.py} + lib/test/parameter/ostrack.py
    (botaoye/OSTrack), stripped of PyTracking's debug/visdom/eval scaffolding.

    ostrack_lib_dir: path to the pruned `lib/` + `experiments/` subset copied
    from the OSTrack repo (see fp_ros_node README section on 2D tracker setup).
    checkpoint_path: path to the released OSTrack_ep0300.pth.tar checkpoint
    (Google Drive link in OSTrack's README) — NOT mae_pretrain_vit_base.pth.
    """

    def __init__(self,
                 ostrack_lib_dir,
                 checkpoint_path,
                 min_score=0.5,
                 yaml_name="vitb_256_mae_ce_32x4_ep300"):
        self.min_score = min_score
        self.initialized = False

        if ostrack_lib_dir not in sys.path:
            sys.path.insert(0, ostrack_lib_dir)
        # Deferred import: only resolvable once ostrack_lib_dir is on sys.path.
        from lib.config.ostrack.config import cfg, update_config_from_file
        from lib.models.ostrack import build_ostrack

        yaml_file = os.path.join(ostrack_lib_dir, "experiments", "ostrack",
                                 f"{yaml_name}.yaml")
        update_config_from_file(yaml_file)
        self.cfg = cfg

        network = build_ostrack(cfg, training=False)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        network.load_state_dict(ckpt["net"], strict=True)
        self.network = network.cuda().eval()

        self.template_factor = cfg.TEST.TEMPLATE_FACTOR
        self.template_size = cfg.TEST.TEMPLATE_SIZE
        self.search_factor = cfg.TEST.SEARCH_FACTOR
        self.search_size = cfg.TEST.SEARCH_SIZE

        feat_sz = cfg.TEST.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE
        self.output_window = _hann2d(feat_sz).cuda()

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

        self.z_dict1 = None
        self.box_mask_z = None
        self.state = None  # xywh, image coords

    def _preprocess(self, img_arr):
        # forward() doesn't consume the attention mask sample_target returns,
        # so only the normalized image tensor is kept (no NestedTensor needed).
        t = torch.tensor(img_arr).cuda().float().permute(2, 0, 1).unsqueeze(0)
        return (t / 255.0 - self.mean) / self.std

    def init(self, rgb, bbox_xyxy):
        from lib.train.data.processing_utils import sample_target
        from lib.utils.ce_utils import generate_mask_cond

        x1, y1, x2, y2 = bbox_xyxy
        init_bbox = [x1, y1, x2 - x1, y2 - y1]  # OSTrack uses xywh internally

        z_patch_arr, _, _ = sample_target(rgb,
                                          init_bbox,
                                          self.template_factor,
                                          output_sz=self.template_size)
        self.z_dict1 = self._preprocess(z_patch_arr)

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            # CE_TEMPLATE_RANGE is 'CTR_POINT' for this config, which only
            # needs (bs, device) — no bbox projection required.
            self.box_mask_z = generate_mask_cond(self.cfg, 1,
                                                 self.z_dict1.device, None)

        self.state = init_bbox
        self.initialized = True

    def track(self, rgb):
        """Returns (bbox_xyxy, score); (None, 0.0) if not initialized."""
        if not self.initialized:
            return None, 0.0

        from lib.train.data.processing_utils import sample_target
        from lib.utils.box_ops import clip_box

        H, W = rgb.shape[:2]
        x_patch_arr, resize_factor, _ = sample_target(
            rgb, self.state, self.search_factor, output_sz=self.search_size)
        search = self._preprocess(x_patch_arr)

        with torch.no_grad():
            out = self.network.forward(template=self.z_dict1,
                                       search=search,
                                       ce_template_mask=self.box_mask_z)

        response = self.output_window * out["score_map"]
        pred_box, score = self.network.box_head.cal_bbox(response,
                                                         out["size_map"],
                                                         out["offset_map"],
                                                         return_score=True)
        cx, cy, w, h = pred_box.view(-1).tolist()
        scale = self.search_size / resize_factor
        pred_box_img = [cx * scale, cy * scale, w * scale, h * scale]

        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        half_side = 0.5 * scale
        cx_r, cy_r, w_r, h_r = pred_box_img
        new_state = [
            cx_r + (cx_prev - half_side) - 0.5 * w_r,
            cy_r + (cy_prev - half_side) - 0.5 * h_r, w_r, h_r
        ]
        self.state = clip_box(new_state, H, W, margin=10)

        x, y, w, h = self.state
        return (x, y, x + w, y + h), float(score.item())


class AngularVelocityKF:
    """Minimal scalar-covariance KF over 3D angular velocity (rad/step),
    used to predict rotation ahead of the refiner each frame."""

    def __init__(self, process_var=0.05, measured_var=0.02, init_var=1.0):
        self.omega = np.zeros(3)
        self.var = init_var
        self.q = process_var
        self.r = measured_var

    def predict(self):
        self.var += self.q
        return self.omega.copy()

    def update(self, measured_omega):
        k = self.var / (self.var + self.r)
        self.omega = self.omega + k * (measured_omega - self.omega)
        self.var = (1 - k) * self.var

    def reset(self):
        self.omega = np.zeros(3)
        self.var = 1.0


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

        # FP++-style prior, split into two independently-toggleable parts:
        #   translation  -> 2D tracker (OSTrack) + depth backprojection
        #   rotation     -> angular-velocity KF (optional; leave off to let the
        #                   refiner handle rotation from the unchanged prev pose)
        # Both off => behaves exactly like the CV-prior path.
        self.use_2d_tracker_translation = False
        self.use_rotation_kf = False
        self._ostrack_lib_dir = f"{os.path.dirname(os.path.realpath(__file__))}/ostrack_lib"
        self._ostrack_checkpoint = f"{self._ostrack_lib_dir}/checkpoints/OSTrack_ep0300.pth.tar"
        self.tracker2d = None  # built lazily by _ensure_tracker2d(), only if enabled
        self.rot_kf = AngularVelocityKF()
        self.kf_rot_prev = None  # rotation at last accepted pose (t-1)

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

        if self.use_2d_tracker_translation:
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            self._ensure_tracker2d().init(rgb, (x, y, x + w, y + h))
        if self.use_rotation_kf:
            self.rot_kf.reset()
            self.kf_rot_prev = self.FPModel.pose_last.detach().cpu().numpy(
            )[:3, :3].copy()

    def _track(self):
        """Frame-to-frame tracking."""
        rgb = self.process_rgb(self.latest_rgb)
        depth = self.process_depth(self.latest_depth)
        cam_K = self.latest_cam_K.copy()

        # Mask-gate depth + seed pose_last (2D-tracker+KF prior, or CV-prior fallback).
        depth = self.gate_depth_by_mask(depth)
        self.apply_2d_kf_prior(rgb, depth, cam_K)

        t0 = time.time()
        pose = self.FPModel.track_one(rgb=rgb,
                                      depth=depth,
                                      K=cam_K,
                                      iteration=self.track_refine_iter)
        elapsed_ms = (time.time() - t0) * 1000
        self.get_logger().info(f"Tracking done in {elapsed_ms:.1f} ms")

        if self.check_auto_reset(pose, cam_K):
            return  # Abort publishing this frame and re-register next frame

        self.update_2d_kf_prior_history()
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

    def _ensure_tracker2d(self):
        """Lazily build Tracker2D on first use, so a missing checkpoint/lib
        dir only breaks things when use_2d_tracker_translation is actually on."""
        if self.tracker2d is None:
            self.tracker2d = Tracker2D(self._ostrack_lib_dir,
                                       self._ostrack_checkpoint,
                                       min_score=0.5)
        return self.tracker2d

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

    # ---------- FP++-style prior: 2D tracker (translation) + KF (rotation) ----------

    def estimate_translation_from_bbox(self, bbox_xyxy, depth, K):
        """Backproject bbox center + median depth inside it to a 3D point.
        Mirrors estimater.guess_translation() but driven by a 2D bbox."""
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        uc, vc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        patch = depth[max(y1, 0):y2, max(x1, 0):x2]
        valid = patch > 0.001
        if not valid.any():
            return None
        zc = np.median(patch[valid])
        center = (np.linalg.inv(K) @ np.array([uc, vc, 1.0]).reshape(3, 1)) * zc
        return center.reshape(3)

    def apply_2d_kf_prior(self, rgb, depth, cam_K):
        """Seed pose_last with independently-toggleable priors:
          translation <- 2D tracker + depth (if use_2d_tracker_translation)
          rotation    <- angular-velocity KF (if use_rotation_kf), else left as
                         the previous pose's rotation for the refiner to handle.
        Falls back to apply_cv_prior() when neither translation nor rotation
        prior is active, or when the 2D tracker is unavailable/low-confidence."""
        if not self.use_2d_tracker_translation and not self.use_rotation_kf:
            self.apply_cv_prior()
            return

        pose_last = self.FPModel.pose_last
        if pose_last is None:
            self.apply_cv_prior()
            return
        cur = pose_last.detach().cpu().numpy().reshape(4, 4)
        pred = cur.copy()

        # --- translation prior: 2D tracker bbox + depth backprojection ---
        if self.use_2d_tracker_translation:
            tracker2d = self._ensure_tracker2d()
            bbox, score = tracker2d.track(rgb)
            if bbox is None or score < tracker2d.min_score:
                self.get_logger().warn(
                    f"2D tracker low-confidence ({score:.2f}) — falling back to CV prior",
                    throttle_duration_sec=2.0)
                self.apply_cv_prior()
                return
            t_2d = self.estimate_translation_from_bbox(bbox, depth, cam_K)
            if t_2d is None:
                self.apply_cv_prior()
                return
            trans_step = np.linalg.norm(t_2d - cur[:3, 3])
            if trans_step > self.max_translation_step:
                self.get_logger().warn(
                    f"2D-tracker jump {trans_step*100:.1f} cm too large — skipping",
                    throttle_duration_sec=2.0)
            else:
                pred[:3, 3] = t_2d.flatten()

        # --- rotation prior: angular-velocity KF (optional) ---
        if self.use_rotation_kf and self.kf_rot_prev is not None:
            omega_pred = self.rot_kf.predict()
            R_pred = R.from_rotvec(omega_pred).as_matrix() @ self.kf_rot_prev
            pred[:3, :3] = R_pred
            u, _, vt = np.linalg.svd(pred[:3, :3])
            pred[:3, :3] = u @ vt

        self.FPModel.pose_last = torch.as_tensor(pred,
                                                 dtype=pose_last.dtype,
                                                 device=pose_last.device)

    def update_2d_kf_prior_history(self):
        """After track_one refines the pose: update the rotation KF with the
        refined rotation as measurement, and keep the CV-prior fallback warm."""
        self.update_cv_prior_history()  # keep fallback path ready regardless
        if not self.use_rotation_kf or self.kf_rot_prev is None:
            return
        pose_last = self.FPModel.pose_last
        if pose_last is None:
            return
        cur_R = pose_last.detach().cpu().numpy().reshape(4, 4)[:3, :3]
        measured_omega = R.from_matrix(cur_R @ self.kf_rot_prev.T).as_rotvec()
        self.rot_kf.update(measured_omega)
        self.kf_rot_prev = cur_R.copy()

    # ---------- auto-reset with centroid distance ----------

    def check_auto_reset(self, pose: np.ndarray, cam_K: np.ndarray) -> bool:
        """
        Detects spatial drift by comparing the SAM2 mask centroid with the
        2D projection of the FoundationPose 3D center. Triggers a reset if lost.
        Returns True if a reset was triggered, False otherwise.
        """
        if not self.use_auto_reset or self.latest_mask is None:
            return False

        mask_bool = self.latest_mask > 0
        if int(mask_bool.sum()) < self.mask_gating_min_pixels:
            return False

        # Compute the 2D centroid of the SAM2 mask
        ys, xs = np.nonzero(mask_bool)
        mask_u, mask_v = float(np.mean(xs)), float(np.mean(ys))

        # Project the FoundationPose 3D center to 2D image space
        t = pose[:3, 3]
        if t[2] <= 0.01:  # Prevent division by zero if behind the camera
            return False

        fp_u = (cam_K[0, 0] * t[0] / t[2]) + cam_K[0, 2]
        fp_v = (cam_K[1, 1] * t[1] / t[2]) + cam_K[1, 2]

        # Calculate Euclidean pixel distance
        dist = np.linalg.norm([mask_u - fp_u, mask_v - fp_v])

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

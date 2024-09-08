#!/usr/bin/env python

import argparse
import logging
import os
import time

import cv2
import numpy as np
import nvdiffrast.torch as dr
import open3d as o3d
import rospy
import trimesh
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Bool

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from Utils import (
    depth2xyzmap,
    draw_posed_3d_box,
    draw_xyz_axis,
    set_logging_format,
    set_seed,
    toOpen3dCloud,
)


class FoundationPoseROS:
    def __init__(self, args):
        set_logging_format()
        set_seed(0)

        # Variables for storing the latest images
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_mask = None
        self.is_object_registered = False

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
        self.reset_sub = rospy.Subscriber(
            "/reset", Bool, self.reset_callback, queue_size=1
        )

        # Publisher for the object pose
        self.pose_pub = rospy.Publisher("/object_pose", Pose, queue_size=1)

    def rgb_callback(self, data):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert RGB image: {e}")

    def depth_callback(self, data):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(data, "64FC1")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert depth image: {e}")

    def mask_callback(self, data):
        try:
            self.latest_mask = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            rospy.logerr(f"Could not convert mask image: {e}")

    def reset_callback(self, data):
        if data.data:
            rospy.loginfo("Resetting the node")
            self.is_object_registered = False
        else:
            rospy.loginfo("Received a reset message with data=False")

    def run(self):
        ##############################
        # Wait for the first images
        ##############################
        while not rospy.is_shutdown() and (
            self.latest_rgb is None
            or self.latest_depth is None
            or self.latest_mask is None
        ):
            rospy.loginfo(
                "Missing one of the required images (RGB, depth, mask). Waiting..."
            )
            rospy.sleep(0.1)

        assert self.latest_rgb is not None
        assert self.latest_depth is not None
        assert self.latest_mask is not None

        while not rospy.is_shutdown():
            if not self.is_object_registered:
                ##############################
                # Run first time
                ##############################
                rospy.loginfo("Running the first frame")

                first_rgb = self.process_rgb(self.latest_rgb)
                first_depth = self.process_depth(self.latest_depth)
                first_mask = self.process_mask(self.latest_mask)

                # Estimation and tracking
                t0 = time.time()
                pose = self.FPModel.register(
                    K=self.cam_K,
                    rgb=first_rgb,
                    depth=first_depth,
                    ob_mask=first_mask,
                    iteration=self.est_refine_iter,
                )
                rospy.loginfo(f"time for reg mask is = {(time.time() - t0)*1000} ms")
                logging.info("First frame estimation done")
                assert pose.shape == (4, 4), f"pose.shape = {pose.shape}"

                if self.debug >= 3:
                    m = self.object_mesh.copy()
                    m.apply_transform(pose)
                    m.export(f"{self.debug_dir}/model_tf.obj")
                    xyz_map = depth2xyzmap(first_depth, self.cam_K)
                    valid = first_depth >= 0.001
                    pcd = toOpen3dCloud(xyz_map[valid], first_rgb[valid])
                    pcd_path = f"{self.debug_dir}/scene_complete.ply"
                    o3d.io.write_point_cloud(pcd_path, pcd)
                    rospy.loginfo(f"Point cloud saved to {pcd_path}")

                self.is_object_registered = True
            else:
                ##############################
                # Track
                ##############################
                start_time = rospy.Time.now()

                rgb = self.process_rgb(self.latest_rgb)
                depth = self.process_depth(self.latest_depth)
                _mask = self.process_mask(self.latest_mask)

                t0 = time.time()
                pose = self.FPModel.track_one(
                    rgb=rgb, depth=depth, K=self.cam_K, iteration=self.track_refine_iter
                )
                rospy.loginfo(f"time for track is = {(time.time() - t0)*1000} ms")

                # Publish pose
                self.publish_pose(pose)

                if self.debug >= 1:
                    center_pose = pose @ np.linalg.inv(self.to_origin)

                    # Must be BGR for cv2
                    vis_img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)

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

                done_time = rospy.Time.now()
                rospy.loginfo(
                    f"Max rate: {np.round(1./(done_time - start_time).to_sec())} Hz ({np.round((done_time - start_time).to_sec()*1000)} ms)"
                )

    def process_rgb(self, rgb):
        rospy.logdebug(f"rgb.shape = {rgb.shape}")
        rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_NEAREST)
        rospy.logdebug(f"AFTER rgb.shape = {rgb.shape}")
        rospy.logdebug(f"rgb: {rgb.shape}, {rgb.dtype}, {np.max(rgb)}, {np.min(rgb)}")
        return rgb

    def process_depth(self, depth):
        rospy.logdebug(f"depth.shape = {depth.shape}")
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)
        rospy.logdebug(f"AFTER depth.shape = {depth.shape}")
        rospy.logdebug(
            f"depth: {depth.shape}, {depth.dtype}, {np.max(depth)}, {np.min(depth)}, {np.mean(depth)}, {np.median(depth)}"
        )
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
        rospy.logdebug(
            f"mask: {mask.shape}, {mask.dtype}, {np.max(mask)}, {np.min(mask)}"
        )
        return mask

    def publish_pose(self, pose: np.ndarray):
        assert pose.shape == (4, 4), f"pose.shape = {pose.shape}"
        trans = pose[:3, 3]
        quat_xyzw = R.from_matrix(pose[:3, :3]).as_quat()

        # Convert the pose matrix into a ROS message
        msg = Pose()
        msg.position.x, msg.position.y, msg.position.z = trans
        msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w = (
            quat_xyzw
        )

        # Publish the pose
        self.pose_pub.publish(msg)
        rospy.logdebug("Pose published to /object_pose")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/kiri_meshes/snackbox/3DModel.obj')
    # parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/kiri_meshes/blueblock/3DModel.obj')
    parser.add_argument(
        "--mesh_file", type=str, default=f"{code_dir}/kiri_meshes/cup_ycbv/textured.obj"
    )
    parser.add_argument("--est_refine_iter", type=int, default=2)
    parser.add_argument("--track_refine_iter", type=int, default=2)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--debug_dir", type=str, default=f"{code_dir}/debug")
    args = parser.parse_args()

    node = FoundationPoseROS(args)
    node.run()

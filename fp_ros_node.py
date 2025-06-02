#!/usr/bin/env python

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
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Int32
from termcolor import colored

from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor
from fp_ros_utils import get_mesh_file
from Utils import (
    depth2xyzmap,
    draw_posed_3d_box,
    draw_xyz_axis,
    set_logging_format,
    set_seed,
    toOpen3dCloud,
)


class FoundationPoseROS:
    def __init__(self):
        set_logging_format()
        set_seed(0)

        # Variables for storing the latest images
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_cam_K = None
        self.latest_mask = None
        self.is_object_registered = False
        self.first = True

        self.first_est_refine_iter = 5  # Can be higher since we only run this once at the very first time to get a good initial pose
        self.est_refine_iter = 1  # Want this as low as possible to run as fast as possible when re-initializing pose (roughly 1 second for each iter?)
        self.track_refine_iter = 2  # Want this as low as possible to run as fast as possible for more accurate tracking (1 seems to be too low though)

        # Debugging
        code_dir = os.path.dirname(os.path.realpath(__file__))
        self.debug = 0  # Keep debug level at 0 to avoid unnecessary overhead since we want this running as fast as possible
        self.debug_dir = f"{code_dir}/debug"
        os.makedirs(self.debug_dir, exist_ok=True)
        os.makedirs(f"{self.debug_dir}/track_vis", exist_ok=True)
        os.makedirs(f"{self.debug_dir}/ob_in_cam", exist_ok=True)

        rospy.init_node("fp_node")
        self.bridge = CvBridge()

        # Load object mesh
        mesh_file = get_mesh_file()
        self.object_mesh = trimesh.load(mesh_file)
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
        print(colored("Estimator initialization done", "green"))

        # Check camera parameter
        camera = rospy.get_param("/camera", None)
        if camera is None:
            DEFAULT_CAMERA = "zed"
            print(
                colored(
                    f"No /camera parameter found, using default camera {DEFAULT_CAMERA}",
                    "yellow",
                )
            )
            camera = DEFAULT_CAMERA
        print(colored(f"Using camera: {camera}", "green"))
        if camera == "zed":
            self.rgb_sub_topic = "/zed/zed_node/rgb/image_rect_color"
            self.depth_sub_topic = "/zed/zed_node/depth/depth_registered"
            self.camera_info_sub_topic = "/zed/zed_node/rgb/camera_info"
        elif camera == "realsense":
            self.rgb_sub_topic = "/camera/color/image_raw"
            self.depth_sub_topic = "/camera/aligned_depth_to_color/image_raw"
            self.camera_info_sub_topic = "/camera/color/camera_info"
        else:
            raise ValueError(f"Unknown camera: {camera}")

        # Subscribers for RGB, depth, and mask images
        self.rgb_sub = rospy.Subscriber(
            self.rgb_sub_topic,
            ROSImage,
            self.rgb_callback,
            queue_size=1,
        )
        self.depth_sub = rospy.Subscriber(
            self.depth_sub_topic,
            # "/depth_anything_v2/depth",
            ROSImage,
            self.depth_callback,
            queue_size=1,
        )
        self.mask_sub = rospy.Subscriber(
            "/sam2_mask", ROSImage, self.mask_callback, queue_size=1
        )
        self.cam_K_sub = rospy.Subscriber(
            self.camera_info_sub_topic,
            CameraInfo,
            self.cam_K_callback,
            queue_size=1,
        )
        self.reset_sub = rospy.Subscriber(
            "/fp_reset", Int32, self.reset_callback, queue_size=1
        )

        # Publisher for the object pose
        self.pose_pub = rospy.Publisher("/object_pose", Pose, queue_size=1)

    def rgb_callback(self, data):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(colored(f"Could not convert RGB image: {e}", "red"))

    def depth_callback(self, data):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(data, "64FC1")
        except CvBridgeError as e:
            print(colored(f"Could not convert depth image: {e}", "red"))

    def mask_callback(self, data):
        try:
            self.latest_mask = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            print(colored(f"Could not convert mask image: {e}", "red"))

    def cam_K_callback(self, data: CameraInfo):
        self.latest_cam_K = np.array(data.K).reshape(3, 3)

    def reset_callback(self, data):
        if data.data > 0:
            print(colored("Resetting the fp node", "green"))
            self.is_object_registered = False
        else:
            print(colored("Received a reset message with data <= 0", "green"))

    def run(self):
        ##############################
        # Wait for the first images
        ##############################
        while not rospy.is_shutdown() and (
            self.latest_rgb is None
            or self.latest_depth is None
            or self.latest_mask is None
            or self.latest_cam_K is None
        ):
            print(
                colored(
                    "Missing one of the required images (RGB, depth, mask, cam_K). Waiting...",
                    "yellow",
                )
            )
            rospy.sleep(0.1)

        assert self.latest_rgb is not None
        assert self.latest_depth is not None
        assert self.latest_mask is not None
        assert self.latest_cam_K is not None

        while not rospy.is_shutdown():
            if not self.is_object_registered:
                ##############################
                # Register
                ##############################
                print(colored("Running registration", "green"))

                register_rgb = self.process_rgb(self.latest_rgb)
                register_depth = self.process_depth(self.latest_depth)
                register_mask = self.process_mask(self.latest_mask)
                register_cam_K = self.latest_cam_K.copy()

                # Estimation and tracking
                t0 = time.time()
                pose = self.FPModel.register(
                    K=register_cam_K,
                    rgb=register_rgb,
                    depth=register_depth,
                    ob_mask=register_mask,
                    iteration=(
                        self.first_est_refine_iter
                        if self.first
                        else self.est_refine_iter
                    ),
                )
                print(
                    colored(
                        f"time for reg mask is = {(time.time() - t0) * 1000} ms",
                        "green",
                    )
                )
                print(colored("Registration done", "green"))
                print(colored(f"pose = {pose}", "green"))
                assert pose.shape == (4, 4), f"pose.shape = {pose.shape}"

                if self.debug >= 3:
                    m = self.object_mesh.copy()
                    m.apply_transform(pose)
                    m.export(f"{self.debug_dir}/model_tf.obj")
                    xyz_map = depth2xyzmap(register_depth, register_cam_K)
                    valid = register_depth >= 0.001
                    pcd = toOpen3dCloud(xyz_map[valid], register_rgb[valid])
                    pcd_path = f"{self.debug_dir}/scene_complete.ply"
                    o3d.io.write_point_cloud(pcd_path, pcd)
                    print(colored(f"Point cloud saved to {pcd_path}", "green"))

                self.is_object_registered = True
                self.first = False
            else:
                ##############################
                # Track
                ##############################
                start_time = rospy.Time.now()

                rgb = self.process_rgb(self.latest_rgb)
                depth = self.process_depth(self.latest_depth)
                _mask = self.process_mask(self.latest_mask)
                cam_K = self.latest_cam_K.copy()

                t0 = time.time()
                pose = self.FPModel.track_one(
                    rgb=rgb, depth=depth, K=cam_K, iteration=self.track_refine_iter
                )
                print(
                    colored(
                        f"time for track is = {(time.time() - t0) * 1000} ms", "green"
                    )
                )

                # Publish pose
                self.publish_pose(pose)

                if self.debug >= 1:
                    center_pose = pose @ np.linalg.inv(self.to_origin)

                    # Must be BGR for cv2
                    vis_img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)

                    vis_img = draw_posed_3d_box(
                        cam_K, img=vis_img, ob_in_cam=center_pose, bbox=self.bbox
                    )
                    vis_img = draw_xyz_axis(
                        vis_img,
                        ob_in_cam=center_pose,
                        scale=0.1,
                        K=cam_K,
                        thickness=3,
                        transparency=0,
                        is_input_rgb=True,
                    )

                    cv2.imshow("Pose Visualization", vis_img)
                    cv2.waitKey(1)

                done_time = rospy.Time.now()
                print(
                    colored(
                        f"Max rate: {np.round(1.0 / (done_time - start_time).to_sec())} Hz ({np.round((done_time - start_time).to_sec() * 1000)} ms)",
                        "green",
                    )
                )

    def process_rgb(self, rgb):
        return rgb

    def process_depth(self, depth):
        # Turn nan values into 0
        depth[np.isnan(depth)] = 0
        depth[np.isinf(depth)] = 0

        # depth is either in meters or millimeters
        # Need to convert to meters
        # If the max value is greater than 100, then it's likely in mm
        in_mm = depth.max() > 100
        if in_mm:
            # print(colored(f"Converting depth from mm to m since max = {depth.max()}", "green"))
            depth = depth / 1000
        else:
            pass
            # print(colored(f"Depth is in meters since max = {depth.max()}", "green"))

        # Clamp
        depth[depth < 0.1] = 0
        depth[depth > 4] = 0

        return depth

    def process_mask(self, mask):
        mask = mask.astype(bool)
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


if __name__ == "__main__":
    node = FoundationPoseROS()
    node.run()

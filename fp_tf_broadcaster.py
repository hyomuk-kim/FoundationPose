#!/usr/bin/env python3
"""
TF broadcaster for the FoundationPose pipeline.

Publishes two transforms:
  1. static:  robot_base -> camera_color_optical_frame  (from calibration npz)
  2. dynamic: camera_color_optical_frame -> fp_object_pose  (from /object_pose)

Together these link the object pose into the robot's TF tree, so the desktop
side can look it up in the world frame without subscribing to the topic.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from scipy.spatial.transform import Rotation
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


class FPTFBroadcaster(Node):

    def __init__(self):
        super().__init__("fp_tf_broadcaster")

        # --- extrinsics (static) ---
        self.declare_parameter(
            "npz_path",
            "camera_036322250488_extrinsics.npz")
        self.declare_parameter("npz_key", "cam2arm")
        self.declare_parameter("robot_frame", "xarm_device")
        self.declare_parameter("camera_frame", "fp_camera_color_optical_frame")

        # --- object pose (dynamic) ---
        self.declare_parameter("pose_topic", "/object_pose")
        self.declare_parameter("object_frame", "fp_object_pose")

        npz_path = self.get_parameter("npz_path").value
        npz_key = self.get_parameter("npz_key").value
        robot_frame = self.get_parameter("robot_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        pose_topic = self.get_parameter("pose_topic").value
        self.object_frame = self.get_parameter("object_frame").value

        # 1. Broadcast the static camera extrinsics once.
        cam2arm = np.load(npz_path)[npz_key]
        quat = Rotation.from_matrix(cam2arm[:3, :3]).as_quat()  # xyzw
        trans = cam2arm[:3, 3]

        static_tf = TransformStamped()
        static_tf.header.stamp = self.get_clock().now().to_msg()
        static_tf.header.frame_id = robot_frame
        static_tf.child_frame_id = self.camera_frame
        static_tf.transform.translation.x = float(trans[0])
        static_tf.transform.translation.y = float(trans[1])
        static_tf.transform.translation.z = float(trans[2])
        static_tf.transform.rotation.x = float(quat[0])
        static_tf.transform.rotation.y = float(quat[1])
        static_tf.transform.rotation.z = float(quat[2])
        static_tf.transform.rotation.w = float(quat[3])

        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.static_broadcaster.sendTransform(static_tf)
        self.get_logger().info(
            f"Static TF: {robot_frame} -> {self.camera_frame}\n"
            f"  translation: {trans}\n  quaternion (xyzw): {quat}")

        # 2. Re-broadcast FP's object pose as a dynamic TF.
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(PoseStamped, pose_topic, self.pose_callback, 1)
        self.get_logger().info(
            f"Bridging {pose_topic} -> TF frame '{self.object_frame}'")

    def pose_callback(self, msg: PoseStamped):
        t = TransformStamped()
        
        # t.header.stamp = msg.header.stamp
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.camera_frame
        t.child_frame_id = self.object_frame

        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = FPTFBroadcaster()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

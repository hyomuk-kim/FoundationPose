import rospy
import numpy as np
import os


def get_mesh_file() -> str:
    # Assumes that rospy.init_node() has been called

    mesh_file = rospy.get_param("/mesh_file", None)
    if mesh_file is None:
        code_dir = os.path.dirname(os.path.realpath(__file__))
        # DEFAULT_MESH_FILE = f"{code_dir}/kiri_meshes/snackbox/3DModel.obj"
        # DEFAULT_MESH_FILE = f"{code_dir}/kiri_meshes/blueblock/3DModel.obj"
        DEFAULT_MESH_FILE = f"{code_dir}/kiri_meshes/cup_ycbv/textured.obj"
        rospy.logwarn(
            f"Mesh file not provided. Using default mesh: {DEFAULT_MESH_FILE}"
        )
        mesh_file = DEFAULT_MESH_FILE
    assert os.path.exists(mesh_file), f"Mesh file does not exist: {mesh_file}"
    return mesh_file


def get_cam_K() -> np.ndarray:
    cam_K_file = rospy.get_param("/cam_K_file", None)
    if cam_K_file is None:
        code_dir = os.path.dirname(os.path.realpath(__file__))
        DEFAULT_CAM_K_FILE = (
            f"{code_dir}/demo_data/blueblock/blueblock_occ_slide/cam_K.txt"
        )
        rospy.logwarn(
            f"Camera intrinsics file not provided. Using default file: {DEFAULT_CAM_K_FILE}"
        )
        cam_K_file = DEFAULT_CAM_K_FILE

    rospy.loginfo(f"cam_K_file = {cam_K_file}")
    cam_K = np.loadtxt(cam_K_file).reshape(3, 3)
    return cam_K

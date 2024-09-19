import os

import numpy as np
import rospy


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
    assert isinstance(mesh_file, str), f"mesh_file: {mesh_file}"
    assert os.path.exists(mesh_file), f"Mesh file does not exist: {mesh_file}"
    return mesh_file
